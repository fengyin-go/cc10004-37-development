"""数据集管理接口：维护数据集定义、字段版本与训练样本。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.schemas import ActionResult, EntryPayload, PageResult
from app.services.datastore import DatastoreService

router = APIRouter(prefix="/api/datastore", tags=["数据集管理"])

service = DatastoreService()

LIST_FIELDS = ["数据集编号", "数据集名称", "来源表名", "字段列表", "更新策略", "数据采样比例", "生命周期天数", "缓存开关"]
STATUSES = ["草稿", "待审核", "已生效", "已废弃"]
ACTIONS = ["提交审核", "生效发布", "废弃下线"]


@router.get("", response_model=PageResult[dict])
def list_entries(
    keyword: str | None = Query(default=None, description="按数据集编号或名称检索"),
    status: str | None = Query(default=None, description="草稿、待审核、已生效、已废弃"),
    page: int = 1,
    size: int = 20,
) -> PageResult[dict]:
    """按数据集编号与状态过滤数据集管理列表；没有数据时返回空页，不报错。"""
    if size > 200:
        raise HTTPException(status_code=400, detail="每页最多 200 条，请缩小分页范围")
    items, total = service.list_entries(keyword=keyword, status=status, page=page, size=size)
    return PageResult(items=items, total=total, page=page, size=size)


@router.get("/export")
def export_entries() -> dict[str, Any]:
    """导出数据集管理清单：返回当前过滤条件下的全量数据。"""
    items, total = service.list_entries(page=1, size=10000)
    return {"module": "datastore", "total": total, "items": items}


@router.get("/lookup")
def lookup_entry(code: str = Query(description="数据集编号，例如 DATA-0001")) -> dict[str, Any]:
    """按数据集编号查定义；证据卡里的字段链接落到这里。"""
    row = service.lookup(code)
    if row is None:
        raise HTTPException(status_code=404, detail=f"数据集 {code} 不存在")
    return row


@router.get("/{entry_id}/versions")
def list_versions(entry_id: int) -> dict[str, Any]:
    """列出字段定义的历史版本；训练签认后仍可对照「当时定义」。"""
    versions = service.list_versions(entry_id)
    if versions is None:
        raise HTTPException(status_code=404, detail=f"数据集定义 {entry_id} 不存在或已归档")
    return {"entry_id": entry_id, "versions": versions}


@router.get("/{entry_id}/versions/{version}")
def get_version(entry_id: int, version: int) -> dict[str, Any]:
    """读取某一版字段定义；被后来修订掉的字段仍按这一版返回。"""
    payload = service.get_version(entry_id, version)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"数据集 {entry_id} 的 v{version} 定义不存在")
    return payload


@router.post("/{entry_id}/revise", response_model=ActionResult)
def revise_entry(entry_id: int, payload: EntryPayload) -> ActionResult:
    """修订字段定义：新增、删除、重命名、改类型、改缺值策略都会追加新版本。"""
    entry, message = service.revise(entry_id, payload.values)
    if entry is None:
        return ActionResult(ok=False, message=message)
    return ActionResult(ok=True, message=message, entry=entry)


@router.put("/{entry_id}/sample", response_model=ActionResult)
def update_sample(entry_id: int, payload: EntryPayload) -> ActionResult:
    """保存可录入的小样本；只接受当前定义里的字段，写入后记一条修订。"""
    entry, message = service.update_sample(entry_id, payload.values)
    if entry is None:
        return ActionResult(ok=False, message=message)
    return ActionResult(ok=True, message=message, entry=entry)


@router.get("/{entry_id}", response_model=dict)
def get_entry(entry_id: int) -> dict:
    """读取单条数据集定义明细，含当前版本字段与样本行。"""
    entry = service.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"数据集定义 {entry_id} 不存在或已归档")
    return entry


@router.post("", response_model=ActionResult)
def create_entry(payload: EntryPayload) -> ActionResult:
    """登记一条数据集定义，缺字段时说明原因而不是静默丢弃。"""
    entry, missing = service.create_entry(payload.values)
    if missing:
        return ActionResult(ok=False, message=f"缺少必填字段：{'、'.join(missing)}")
    return ActionResult(ok=True, message="数据集定义已登记", entry=entry)


@router.post("/{entry_id}/actions", response_model=ActionResult)
def run_action(entry_id: int, payload: EntryPayload) -> ActionResult:
    """对单条数据集定义执行提交审核、生效发布、废弃下线；不允许的动作会被拦下并说明原因。"""
    action = str(payload.values.get("action") or "").strip()
    entry, message = service.run_action(entry_id, action)
    if entry is None:
        return ActionResult(ok=False, message=message)
    return ActionResult(ok=True, message=message, entry=entry)
