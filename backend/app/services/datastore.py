"""数据集管理业务规则：状态流转、字段版本、样本修订与筛选口径都收在这里。

字段定义按版本追加保存，样本行另记修订。趋势预测发起训练时只深拷贝当刻的
定义版本与样本，所以这里的修订、删列、改名都不会回头改写已签认的训练证据卡。
"""
from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from app.services.lock import STORE_LOCK
from app.store import store

MODULE = "datastore"
REQUIRED_FIELDS = ["数据集编号", "数据集名称", "来源表名"]
STATUS_ORDER = ["草稿", "待审核", "已生效", "已废弃"]
ACTION_RULES = {"提交审核": "待审核", "生效发布": "已生效", "废弃下线": "已废弃"}
NEGATIVE_ACTIONS = []
FIELD_TYPES = ["文本", "数值", "日期", "布尔"]
MISSING_POLICIES = ["丢弃该行", "均值填充", "中位数填充", "众数填充", "零值填充", "不填充"]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _fields_text(fields: list[dict[str, Any]]) -> str:
    return "、".join(str(item.get("name")) for item in fields if item.get("name"))


class DatastoreService:
    def list_entries(
        self,
        *,
        keyword: str | None = None,
        status: str | None = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        rows = store.rows(MODULE)
        if keyword:
            rows = [
                row for row in rows
                if keyword in str(row.get("数据集编号", "")) or keyword in str(row.get("数据集名称", ""))
            ]
        if status:
            rows = [row for row in rows if row.get("status") == status]
        total = len(rows)
        start = max(page - 1, 0) * size
        return [self._public(row) for row in rows[start:start + size]], total

    def get_entry(self, entry_id: int) -> dict[str, Any] | None:
        row = store.find(MODULE, entry_id)
        return self._public(row) if row else None

    def lookup(self, code: str) -> dict[str, Any] | None:
        for row in store.rows(MODULE):
            if str(row.get("数据集编号")) == code:
                return self._public(row)
        return None

    def _public(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = copy.deepcopy(row)
        versions = payload.get("定义版本") or []
        payload["current_version"] = int(versions[-1].get("version", 0)) if versions else 0
        payload["version_count"] = len(versions)
        payload["sample_revision"] = len(payload.get("样本修订记录") or [])
        return payload

    def create_entry(self, values: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        missing = [field for field in REQUIRED_FIELDS if not str(values.get(field) or "").strip()]
        if missing:
            return None, missing
        with STORE_LOCK:
            rows = store.rows(MODULE)
            entry = {"id": max((int(row.get("id", 0)) for row in rows), default=0) + 1}
            entry.update({field: values.get(field) for field in REQUIRED_FIELDS})
            entry["字段列表"] = ""
            entry["status"] = STATUS_ORDER[0]
            entry["pending"] = True
            entry["abnormal"] = False
            entry["定义版本"] = [{
                "version": 1,
                "created_at": _now(),
                "operator": values.get("operator") or "值班管理员",
                "note": "登记数据集，字段定义从空白开始",
                "source_table": entry.get("来源表名"),
                "fields": [],
            }]
            entry["样本数据"] = []
            entry["样本说明"] = ""
            entry["样本修订记录"] = []
            rows.append(entry)
            return self._public(entry), []

    def run_action(self, entry_id: int, action: str) -> tuple[dict[str, Any] | None, str]:
        entry = store.find(MODULE, entry_id)
        if entry is None:
            return None, f"数据集定义 {entry_id} 不存在或已归档"
        if action not in ACTION_RULES:
            return None, f"动作「{action}」不属于数据集管理可执行范围"
        target = ACTION_RULES[action]
        if target not in STATUS_ORDER:
            return None, f"目标状态「{target}」不在允许的状态序列里"
        entry["status"] = target
        entry["pending"] = target != STATUS_ORDER[-1]
        entry["abnormal"] = action in NEGATIVE_ACTIONS
        return self._public(entry), f"数据集定义已{action}"

    # ---------------------------------------------------------------- 字段版本
    def list_versions(self, entry_id: int) -> list[dict[str, Any]] | None:
        row = store.find(MODULE, entry_id)
        if row is None:
            return None
        return [
            {
                "version": item.get("version"),
                "created_at": item.get("created_at"),
                "operator": item.get("operator"),
                "note": item.get("note"),
                "source_table": item.get("source_table"),
                "field_count": len(item.get("fields") or []),
                "fields": copy.deepcopy(item.get("fields") or []),
            }
            for item in row.get("定义版本") or []
        ]

    def get_version(self, entry_id: int, version: int) -> dict[str, Any] | None:
        row = store.find(MODULE, entry_id)
        if row is None:
            return None
        for item in row.get("定义版本") or []:
            if int(item.get("version", 0)) == version:
                return copy.deepcopy(item)
        return None

    def revise(self, entry_id: int, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        action = str(payload.get("action") or "").strip()
        operator = str(payload.get("operator") or "值班管理员").strip() or "值班管理员"
        note = str(payload.get("note") or "").strip()
        with STORE_LOCK:
            row = store.find(MODULE, entry_id)
            if row is None:
                return None, f"数据集定义 {entry_id} 不存在或已归档"
            versions = row.get("定义版本") or []
            if not versions:
                return None, "该数据集没有可修订的字段定义版本"
            current = versions[-1]
            fields = copy.deepcopy(current.get("fields") or [])
            by_name = {str(item.get("name")): item for item in fields}

            if action == "新增字段":
                name = str(payload.get("name") or "").strip()
                if not name:
                    return None, "新增字段时必须提供字段名"
                if name in by_name:
                    return None, f"字段 {name} 已存在于 v{current.get('version')}，请改用修改字段"
                field_type = str(payload.get("type") or "文本")
                if field_type not in FIELD_TYPES:
                    return None, f"字段类型「{field_type}」不在允许范围：{'、'.join(FIELD_TYPES)}"
                policy = str(payload.get("missing_policy") or "不填充")
                if policy not in MISSING_POLICIES:
                    return None, f"缺值策略「{policy}」不在允许范围：{'、'.join(MISSING_POLICIES)}"
                fields.append({
                    "name": name,
                    "type": field_type,
                    "nullable": bool(payload.get("nullable", True)),
                    "missing_policy": policy,
                    "description": str(payload.get("description") or "").strip(),
                })
                change = f"新增字段 {name}"
                for sample in row.get("样本数据") or []:
                    sample.setdefault(name, None)
            elif action == "删除字段":
                name = str(payload.get("name") or "").strip()
                if name not in by_name:
                    return None, f"字段 {name} 不在 v{current.get('version')} 的定义里"
                if len(fields) <= 1:
                    return None, "至少保留一个字段，不能删除最后一个字段"
                fields = [item for item in fields if str(item.get("name")) != name]
                for sample in row.get("样本数据") or []:
                    sample.pop(name, None)
                change = f"删除字段 {name}"
            elif action == "重命名字段":
                old = str(payload.get("name") or "").strip()
                new = str(payload.get("new_name") or "").strip()
                if old not in by_name:
                    return None, f"字段 {old} 不在 v{current.get('version')} 的定义里"
                if not new:
                    return None, "重命名字段时必须提供新字段名"
                if new in by_name:
                    return None, f"新字段名 {new} 已存在，不能重名"
                by_name[old]["name"] = new
                for sample in row.get("样本数据") or []:
                    if old in sample:
                        sample[new] = sample.pop(old)
                change = f"字段 {old} 重命名为 {new}"
            elif action == "修改缺值策略":
                name = str(payload.get("name") or "").strip()
                if name not in by_name:
                    return None, f"字段 {name} 不在 v{current.get('version')} 的定义里"
                policy = str(payload.get("missing_policy") or "")
                if policy not in MISSING_POLICIES:
                    return None, f"缺值策略「{policy}」不在允许范围：{'、'.join(MISSING_POLICIES)}"
                by_name[name]["missing_policy"] = policy
                change = f"字段 {name} 的缺值策略改为 {policy}"
            elif action == "修改字段类型":
                name = str(payload.get("name") or "").strip()
                if name not in by_name:
                    return None, f"字段 {name} 不在 v{current.get('version')} 的定义里"
                field_type = str(payload.get("type") or "")
                if field_type not in FIELD_TYPES:
                    return None, f"字段类型「{field_type}」不在允许范围：{'、'.join(FIELD_TYPES)}"
                by_name[name]["type"] = field_type
                change = f"字段 {name} 的类型改为 {field_type}"
            else:
                return None, f"修订动作「{action}」不属于数据集管理可执行范围"

            new_version = int(current.get("version", 0)) + 1
            versions.append({
                "version": new_version,
                "created_at": _now(),
                "operator": operator,
                "note": note or change,
                "source_table": row.get("来源表名"),
                "fields": fields,
            })
            row["定义版本"] = versions
            row["字段列表"] = _fields_text(fields)
            return self._public(row), f"已生成 v{new_version}：{change}（{len(fields)} 个字段）"

    # ---------------------------------------------------------------- 样本修订
    def update_sample(self, entry_id: int, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        operator = str(payload.get("operator") or "值班管理员").strip() or "值班管理员"
        note = str(payload.get("note") or "").strip()
        rows_input = payload.get("rows")
        if rows_input is None:
            return None, "保存样本需要提供 rows 数组"
        if not isinstance(rows_input, list):
            return None, "rows 必须是样本行的数组"
        if len(rows_input) > 200:
            return None, "单次最多保存 200 行样本，便于逐格核对"
        with STORE_LOCK:
            row = store.find(MODULE, entry_id)
            if row is None:
                return None, f"数据集定义 {entry_id} 不存在或已归档"
            versions = row.get("定义版本") or []
            fields = (versions[-1].get("fields") or []) if versions else []
            allowed = {str(item.get("name")): item for item in fields}
            normalized: list[dict[str, Any]] = []
            for index, item in enumerate(rows_input, start=1):
                if not isinstance(item, dict):
                    return None, f"第 {index} 行样本不是对象"
                extra = [key for key in item.keys() if key not in allowed]
                if extra:
                    return None, (
                        f"第 {index} 行出现当前定义没有的字段：{'、'.join(extra)}；"
                        "请先在字段定义中新增，避免写入无法解释的列"
                    )
                sample: dict[str, Any] = {}
                for name, definition in allowed.items():
                    raw = item.get(name)
                    sample[name] = self._coerce(raw, str(definition.get("type") or ""))
                normalized.append(sample)
            before = copy.deepcopy(row.get("样本数据") or [])
            revision = len(row.get("样本修订记录") or []) + 1
            row.setdefault("样本修订记录", []).append({
                "revision": revision,
                "updated_at": _now(),
                "operator": operator,
                "note": note or f"替换为 {len(normalized)} 行样本",
                "before": before,
                "after": copy.deepcopy(normalized),
            })
            row["样本数据"] = normalized
            return self._public(row), f"样本已保存为修订 #{revision}（{len(normalized)} 行）；下一轮训练将使用这份样本"

    def _coerce(self, raw: Any, field_type: str) -> Any:
        if raw is None:
            return None
        if isinstance(raw, str) and not raw.strip():
            return None
        if field_type == "数值":
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return raw
            try:
                number = float(str(raw).strip())
            except (TypeError, ValueError):
                return str(raw).strip()
            return int(number) if number == int(number) else number
        if field_type == "布尔":
            if isinstance(raw, bool):
                return 1 if raw else 0
            text = str(raw).strip().lower()
            if text in {"1", "true", "是", "开启"}:
                return 1
            if text in {"0", "false", "否", "关闭"}:
                return 0
            return text
        return raw if isinstance(raw, str) else str(raw)
