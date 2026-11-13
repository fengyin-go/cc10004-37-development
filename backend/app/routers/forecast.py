"""趋势预测接口：维护预测模型，并承载训练轮次、证据解释与签认。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.schemas import ActionResult, EntryPayload, PageResult
from app.services.forecast import ForecastService
from app.services.forecast_training import training_service
from app.store import store

router = APIRouter(prefix="/api/forecast", tags=["趋势预测"])

service = ForecastService()

LIST_FIELDS = ["模型编号", "模型类型", "训练数据集", "目标字段", "特征字段", "预测步长", "置信区间", "评估指标"]
STATUSES = ["草稿", "待训练", "已生效", "已失效"]


@router.get("", response_model=PageResult[dict])
def list_entries(
    keyword: str | None = Query(default=None, description="按模型编号检索"),
    status: str | None = Query(default=None, description="草稿、待训练、已生效、已失效"),
    page: int = 1,
    size: int = 20,
) -> PageResult[dict]:
    """按模型编号与状态过滤趋势预测列表；没有数据时返回空页，不报错。"""
    if size > 200:
        raise HTTPException(status_code=400, detail="每页最多 200 条，请缩小分页范围")
    items, total = service.list_entries(keyword=keyword, status=status, page=page, size=size)
    return PageResult(items=items, total=total, page=page, size=size)


@router.get("/export")
def export_entries() -> dict[str, Any]:
    """导出趋势预测清单：返回当前过滤条件下的全量数据。"""
    items, total = service.list_entries(page=1, size=10000)
    return {"module": "forecast", "total": total, "items": items}


@router.get("/{entry_id}/rounds")
def list_rounds(entry_id: int) -> dict[str, Any]:
    """列出某模型的训练轮次：当前评估位、迟到轮次与已签认轮次一览。"""
    entry, payload = training_service.list_rounds(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"预测模型 {entry_id} 不存在或已归档")
    return payload


@router.post("/{entry_id}/rounds", response_model=ActionResult)
def start_round(entry_id: int, payload: EntryPayload) -> ActionResult:
    """发起一轮训练：在锁内深拷贝数据集当前定义版本与样本，形成冻结快照。

    values 支持 note（轮次备注）、operator、delay_ms（模拟解释迟到）、defer（先冻结快照稍后生成解释）。
    """
    note = str(payload.values.get("note") or payload.remark or "").strip()
    operator = str(payload.values.get("operator") or "值班管理员").strip() or "值班管理员"
    try:
        delay_ms = int(payload.values.get("delay_ms") or 0)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="delay_ms 需要是整数毫秒")
    defer = bool(payload.values.get("defer"))
    round_, message = training_service.start_round(
        entry_id,
        note=note,
        operator=operator,
        delay_ms=delay_ms,
        defer=defer,
    )
    if round_ is None:
        return ActionResult(ok=False, message=message)
    return ActionResult(ok=True, message=message, entry=round_)


@router.get("/{entry_id}/rounds/current")
def get_current_round(entry_id: int) -> dict[str, Any]:
    """读取当前轮次（最新发起的轮次）；没有轮次时返回空标记而不是 404。"""
    entry = store.find("forecast", entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"预测模型 {entry_id} 不存在或已归档")
    payload = training_service.current_round(entry_id)
    if payload is None:
        return {"model": entry, "summary": None, "message": "该模型还没有训练轮次，可先发起一轮训练。"}
    payload["model"] = entry
    return payload


@router.get("/{entry_id}/rounds/{round_id}")
def get_round(entry_id: int, round_id: int) -> dict[str, Any]:
    """读取一轮训练的快照、四张证据卡、签认状态与数据集漂移提示。"""
    payload = training_service.get_round(entry_id, round_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"训练轮次 {round_id} 不存在")
    return payload


@router.post("/{entry_id}/rounds/{round_id}/explain", response_model=ActionResult)
def explain_round(entry_id: int, round_id: int, payload: EntryPayload) -> ActionResult:
    """为已冻结的轮次生成解释；已经生成或已签认时幂等返回原结果。"""
    operator = str(payload.values.get("operator") or "解释器").strip() or "解释器"
    round_, message = training_service.explain(entry_id, round_id, operator=operator)
    if round_ is None:
        return ActionResult(ok=False, message=message)
    return ActionResult(ok=True, message=message, entry=round_)


@router.post("/{entry_id}/rounds/{round_id}/sign", response_model=ActionResult)
def sign_round(entry_id: int, round_id: int, payload: EntryPayload) -> ActionResult:
    """签认一轮训练：原子封存输入版本与解释。重复签认幂等返回同一封存结果。"""
    operator = str(payload.values.get("operator") or "值班管理员").strip() or "值班管理员"
    note = str(payload.values.get("note") or payload.remark or "").strip()
    round_, message = training_service.sign_off(entry_id, round_id, operator=operator, note=note)
    if round_ is None:
        return ActionResult(ok=False, message=message)
    return ActionResult(ok=True, message=message, entry=round_)


@router.get("/{entry_id}", response_model=dict)
def get_entry(entry_id: int) -> dict:
    """读取单条预测模型明细；不存在时给出可读的错误说明。"""
    entry = service.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"预测模型 {entry_id} 不存在或已归档")
    return entry


@router.post("", response_model=ActionResult)
def create_entry(payload: EntryPayload) -> ActionResult:
    """登记一条预测模型，缺字段时说明原因而不是静默丢弃。"""
    entry, missing = service.create_entry(payload.values)
    if missing:
        return ActionResult(ok=False, message=f"缺少必填字段：{'、'.join(missing)}")
    return ActionResult(ok=True, message="预测模型已登记", entry=entry)


@router.patch("/{entry_id}", response_model=ActionResult)
def update_entry(entry_id: int, payload: EntryPayload) -> ActionResult:
    """修改模型的训练配置；历史轮次仍引用签认时的配置快照。"""
    entry, message = service.update_entry(entry_id, payload.values)
    if entry is None:
        return ActionResult(ok=False, message=message)
    return ActionResult(ok=True, message=message, entry=entry)


@router.post("/{entry_id}/actions", response_model=ActionResult)
def run_action(entry_id: int, payload: EntryPayload) -> ActionResult:
    """对单条预测模型执行提交训练、生效发布、标记失效；不允许的动作会被拦下并说明原因。"""
    action = str(payload.values.get("action") or "").strip()
    entry, message = service.run_action(entry_id, action)
    if entry is None:
        return ActionResult(ok=False, message=message)
    return ActionResult(ok=True, message=message, entry=entry)
