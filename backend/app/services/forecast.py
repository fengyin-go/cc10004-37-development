"""趋势预测业务规则：状态流转、字段校验与筛选口径都收在这里。"""
from __future__ import annotations

from typing import Any

from app.store import store

MODULE = "forecast"
REQUIRED_FIELDS = ["模型编号", "模型类型", "训练数据集"]
OPTIONAL_FIELDS = ["目标字段", "特征字段", "预测步长", "置信区间", "评估指标"]
STATUS_ORDER = ["草稿", "待训练", "已生效", "已失效"]
ACTION_RULES = {"提交训练": "待训练", "生效发布": "已生效", "标记失效": "已失效"}
NEGATIVE_ACTIONS = []


class ForecastService:
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
            rows = [row for row in rows if keyword in str(row.get("模型编号", ""))]
        if status:
            rows = [row for row in rows if row.get("status") == status]
        total = len(rows)
        start = max(page - 1, 0) * size
        return rows[start:start + size], total

    def get_entry(self, entry_id: int) -> dict[str, Any] | None:
        return store.find(MODULE, entry_id)

    def create_entry(self, values: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        missing = [field for field in REQUIRED_FIELDS if not str(values.get(field) or "").strip()]
        if missing:
            return None, missing
        rows = store.rows(MODULE)
        entry = {"id": max((int(row.get("id", 0)) for row in rows), default=0) + 1}
        entry.update({field: values.get(field) for field in REQUIRED_FIELDS})
        for field in OPTIONAL_FIELDS:
            entry[field] = str(values.get(field) or "").strip()
        entry["status"] = STATUS_ORDER[0]
        entry["pending"] = True
        entry["abnormal"] = False
        rows.append(entry)
        return entry, []

    def update_entry(self, entry_id: int, values: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        """修改训练配置（目标字段、特征、评估指标等）；下一轮训练按新配置解释。"""
        entry = store.find(MODULE, entry_id)
        if entry is None:
            return None, f"预测模型 {entry_id} 不存在或已归档"
        for field in REQUIRED_FIELDS:
            if field not in values:
                continue
            text = str(values.get(field) or "").strip()
            if not text:
                return None, f"必填字段「{field}」不能为空"
            entry[field] = text
        for field in OPTIONAL_FIELDS:
            if field in values:
                entry[field] = str(values.get(field) or "").strip()
        return entry, "预测模型配置已更新，下一轮训练将按新配置解释"

    def run_action(self, entry_id: int, action: str) -> tuple[dict[str, Any] | None, str]:
        entry = store.find(MODULE, entry_id)
        if entry is None:
            return None, f"预测模型 {entry_id} 不存在或已归档"
        if action not in ACTION_RULES:
            return None, f"动作「{action}」不属于趋势预测可执行范围"
        target = ACTION_RULES[action]
        if target not in STATUS_ORDER:
            return None, f"目标状态「{target}」不在允许的状态序列里"
        entry["status"] = target
        entry["pending"] = target != STATUS_ORDER[-1]
        entry["abnormal"] = action in NEGATIVE_ACTIONS
        return entry, f"预测模型已{action}"
