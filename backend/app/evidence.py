"""训练证据解释引擎：目标字段 → 特征取值 → 缺值处理 → 评估指标。

引擎只接受一份「训练轮次快照」：数据集某一版字段定义、快照样本、模型配置。
它不会回查数据集管理的当前定义，因此已经签认的轮次不会被后来的删列或改名改写；
快照里没有的定义也不会用今天的定义补造。

每个卡片都带三样东西：由样本实算出的指标、可展开核对的逐行证据、以及
无法解释时的明确留空与依据。
"""
from __future__ import annotations

from datetime import date
from typing import Any

from app.evidence_stats import (
    as_number,
    auc_score,
    binary_labels,
    category_stats,
    display,
    fit_linear,
    is_blank,
    num_text,
    numeric_stats,
    pearson,
    predict,
    quantile,
    regression_metrics,
)

STEP_TARGET = 1
STEP_FEATURE = 2
STEP_MISSING = 3
STEP_METRIC = 4

CARD_TITLES = {
    STEP_TARGET: "目标字段",
    STEP_FEATURE: "特征取值",
    STEP_MISSING: "缺值处理",
    STEP_METRIC: "评估指标",
}

REGRESSION_METRICS = {"MAE", "RMSE", "MAPE"}
CLASSIFICATION_METRICS = {"AUC", "F1", "准确率", "ACC", "ACCURACY"}
POSITIVE_WORDS = {"1", "true", "是", "已退款", "退款", "流失", "转化"}

STATUS_LABELS = {
    "explained": "口径完整",
    "partial": "部分留空",
    "missing": "缺失定义",
}


def _same_cell(left: Any, right: Any) -> bool:
    return display(left) == display(right)


class EvidenceEngine:
    """把一份冻结快照解释成四张证据卡。"""

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        self.dataset: dict[str, Any] = snapshot.get("dataset") or {}
        self.model: dict[str, Any] = snapshot.get("model") or {}
        self.rows: list[dict[str, Any]] = snapshot.get("sample") or []
        self.fields: list[dict[str, Any]] = list(self.dataset.get("fields") or [])
        self.defs: dict[str, dict[str, Any]] = {
            str(field.get("name")): field for field in self.fields if field.get("name")
        }
        self.target: str = str(self.model.get("target") or "").strip()
        self.features: list[str] = [
            str(item).strip() for item in (self.model.get("features") or []) if str(item).strip()
        ]
        self.metric: str = str(self.model.get("metric") or "").strip()
        self.target_def = self.defs.get(self.target) if self.target else None
        self.fill_values: dict[str, Any] = {}
        self.effective: list[dict[str, Any]] = []
        self.row_drops: dict[int, str] = {}
        self.field_notes: dict[str, list[dict[str, str]]] = {}
        self.analysis: dict[str, dict[str, Any]] = {}
        self._usable_rows: list[int] = []
        self._fit_summary: dict[str, Any] = {"metric": self.metric, "value": None}

    # ------------------------------------------------------------------ 公共入口
    def build(self) -> dict[str, Any]:
        self._prepare_rows()
        cards = [
            self._target_card(),
            self._feature_card(),
            self._missing_card(),
            self._metric_card(),
        ]
        gaps = [
            {"card": card["key"], "card_title": card["title"], "field": item.get("field", ""), "reason": item["text"]}
            for card in cards
            for item in card.get("basis", [])
            if item.get("kind") == "missing"
        ]
        metric_values = {
            item["label"]: item["value"]
            for item in cards[3].get("metrics", [])
            if item.get("kind") in {"metric", "baseline"}
        }
        return {
            "cards": cards,
            "gaps": gaps,
            "metric_values": metric_values,
            "fit": self._fit_summary,
        }

    # ------------------------------------------------------------------ 预处理
    def _prepare_rows(self) -> None:
        needed = list(dict.fromkeys(([self.target] if self.target else []) + self.features))
        self.analysis = {name: self._field_analysis(name) for name in needed}
        self.field_notes = {name: info["notes"] for name, info in self.analysis.items()}
        self.fill_values = {
            name: info["fill_value"] for name, info in self.analysis.items() if info["fill_value"] is not None
        }
        for index, row in enumerate(self.rows):
            effective: dict[str, Any] = {}
            for name, info in self.analysis.items():
                raw = row.get(name)
                if is_blank(raw) and info["policy"] == "丢弃该行":
                    self.row_drops.setdefault(index, f"{name} 为空且缺值策略为丢弃该行")
                    effective[name] = None
                elif is_blank(raw) and info["fill_value"] is not None:
                    effective[name] = info["fill_value"]
                else:
                    effective[name] = raw
            self.effective.append(effective)

    def _field_analysis(self, name: str) -> dict[str, Any]:
        definition = self.defs.get(name)
        raw_values = [row.get(name) for row in self.rows]
        missing = sum(1 for value in raw_values if is_blank(value))
        policy = str(definition.get("missing_policy") or "").strip() if definition else ""
        notes: list[dict[str, str]] = []
        fill_value: Any = None
        fill_kind = ""
        failed = ""
        numeric_values = [number for number in (as_number(value) for value in raw_values) if number is not None]

        if definition is None:
            if missing:
                notes.append({
                    "kind": "missing",
                    "field": name,
                    "text": (
                        f"快照字段定义中没有 {name}；样本有 {missing} 行为空，无法按当时的缺值策略处理，"
                        "未用当前定义补造。"
                    ),
                })
        elif not policy:
            notes.append({
                "kind": "missing",
                "field": name,
                "text": f"{name} 在 v{self.dataset.get('version')} 定义里没有登记缺值策略，空值无法解释处理方式。",
            })
        elif policy == "丢弃该行":
            if missing:
                notes.append({"kind": "note", "field": name, "text": f"{name} 有 {missing} 行为空，按定义整行退出训练。"})
        elif policy in {"均值填充", "中位数填充"}:
            if not numeric_values:
                if missing:
                    failed = f"{name} 没有可用的非空数值，{policy}无法计算，空值保持留空。"
                    notes.append({"kind": "missing", "field": name, "text": failed})
            else:
                if policy == "均值填充":
                    fill_value = sum(numeric_values) / len(numeric_values)
                    fill_kind = "均值"
                else:
                    fill_value = quantile(numeric_values, 0.5)
                    fill_kind = "中位数"
                notes.append({
                    "kind": "note",
                    "field": name,
                    "text": (
                        f"{name} 按{policy}填入 {num_text(fill_value)}"
                        f"（由 {len(numeric_values)} 个非空数值算出）。"
                    ),
                })
        elif policy == "众数填充":
            categories = category_stats([value for value in raw_values if not is_blank(value)], limit=50)
            if categories:
                fill_value = categories[0]["value"]
                fill_kind = "众数"
                notes.append({
                    "kind": "note",
                    "field": name,
                    "text": (
                        f"{name} 按众数填入「{fill_value}」，"
                        f"在 {len(categories)} 个非空取值里出现 {categories[0]['count']} 次。"
                    ),
                })
            elif missing:
                failed = f"{name} 没有可用的非空取值，众数无法计算，空值保持留空。"
                notes.append({"kind": "missing", "field": name, "text": failed})
        elif policy == "零值填充":
            fill_value = 0
            fill_kind = "零值"
            notes.append({"kind": "note", "field": name, "text": f"{name} 按零值填充，共 {missing} 行受影响。"})
        elif policy == "不填充":
            if missing:
                notes.append({
                    "kind": "note",
                    "field": name,
                    "text": f"{name} 策略为不填充，{missing} 行空值原样保留，不参与特征拟合。",
                })
        return {
            "name": name,
            "defined": definition is not None,
            "definition": definition,
            "policy": policy,
            "type": (definition or {}).get("type") or "",
            "raw_missing": missing,
            "fill_value": fill_value,
            "fill_kind": fill_kind,
            "failed": failed,
            "notes": notes,
            "numeric_values": numeric_values,
            "raw_values": raw_values,
        }

    # ------------------------------------------------------------------ 卡片工具
    def _definition_ref(self, name: str) -> dict[str, Any]:
        base = {
            "dataset_id": self.dataset.get("id"),
            "dataset_code": self.dataset.get("code"),
            "dataset_name": self.dataset.get("name"),
            "version": self.dataset.get("version"),
            "version_created_at": self.dataset.get("version_created_at"),
            "field": name,
            "link": (
                f"/datastore?dataset={self.dataset.get('code')}"
                f"&version={self.dataset.get('version')}&field={name}"
            ),
        }
        definition = self.defs.get(name)
        if definition is None:
            base.update({"present": False, "type": "", "nullable": None, "description": "", "missing_policy": ""})
            return base
        base.update({
            "present": True,
            "type": definition.get("type") or "",
            "nullable": definition.get("nullable"),
            "description": definition.get("description") or "",
            "missing_policy": definition.get("missing_policy") or "",
        })
        return base

    def _row_columns(self, names: list[str]) -> list[dict[str, Any]]:
        columns = [{"key": "row_index", "label": "样本行", "type": "index"}]
        for name in names:
            definition = self.defs.get(name)
            columns.append({
                "key": name,
                "label": name,
                "type": "number" if (definition or {}).get("type") in {"数值", "布尔"} else "text",
            })
        columns.append({"key": "_included", "label": "参与训练", "type": "badge"})
        columns.append({"key": "_reason", "label": "说明", "type": "text"})
        return columns

    def _evidence_rows(self, names: list[str], included: dict[int, bool] | None = None) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for index, row in enumerate(self.rows):
            in_fit = True if included is None else included.get(index, False)
            cells: dict[str, Any] = {"row_index": index + 1}
            for name in names:
                cells[name] = display(row.get(name))
            cells["_included"] = "是" if in_fit else "否"
            cells["_reason"] = "" if in_fit else self.row_drops.get(index, "该行不满足训练条件")
            items.append({"row_index": index, "cells": cells, "included": in_fit})
        return {"columns": self._row_columns(names), "items": items}

    def _card_base(self, step: int, key: str, status: str, summary: str) -> dict[str, Any]:
        return {
            "key": key,
            "step": step,
            "title": CARD_TITLES[step],
            "status": status,
            "status_text": STATUS_LABELS[status],
            "summary": summary,
            "definitions": [],
            "metrics": [],
            "basis": [],
            "rows": {"columns": [], "items": []},
            "extras": {},
        }

    # ------------------------------------------------------------------ 目标字段
    def _target_card(self) -> dict[str, Any]:
        definitions = [self._definition_ref(self.target)] if self.target else []
        basis: list[dict[str, str]] = []
        status = "explained"
        if not self.target:
            card = self._card_base(STEP_TARGET, "target", "missing", "留空：模型没有配置目标字段。")
            card["basis"] = [{"kind": "missing", "field": "", "text": "模型配置没有登记目标字段，本轮无法解释目标口径。"}]
            return card
        if self.target_def is None:
            status = "missing"
            basis.append({
                "kind": "missing",
                "field": self.target,
                "text": (
                    f"数据集 {self.dataset.get('code') or '—'} 的 v{self.dataset.get('version') or '—'} 定义里没有 "
                    f"{self.target}；快照样本里这一列的值只作历史记录展示，不用今天的数据集定义补造口径。"
                ),
            })
        if self.rows and self.target not in self.rows[0]:
            basis.append({"kind": "note", "field": self.target, "text": f"快照样本没有采集到 {self.target} 列。"})

        values = [row.get(self.target) for row in self.rows]
        missing = sum(1 for value in values if is_blank(value))
        numbers = [number for number in (as_number(value) for value in values) if number is not None]
        categories = category_stats([value for value in values if not is_blank(value)])
        target_type = (self.target_def or {}).get("type") or ""

        metrics = [
            {"label": "样本行数", "value": str(len(self.rows)), "kind": "stat"},
            {"label": "非空目标", "value": str(len(self.rows) - missing), "kind": "stat"},
            {"label": "目标为空", "value": str(missing), "kind": "stat"},
        ]
        extras: dict[str, Any] = {"target_type": target_type, "missing": missing}
        if numbers:
            stats = numeric_stats(numbers)
            extras["numeric"] = {
                "count": stats["count"],
                "mean": num_text(stats["mean"]),
                "median": num_text(stats["median"]),
                "min": num_text(stats["min"]),
                "max": num_text(stats["max"]),
            }
            metrics.extend([
                {"label": "均值", "value": num_text(stats["mean"]), "kind": "stat"},
                {"label": "中位数", "value": num_text(stats["median"]), "kind": "stat"},
                {"label": "取值区间", "value": f"{num_text(stats['min'])} ~ {num_text(stats['max'])}", "kind": "stat"},
            ])
            summary = (
                f"{self.target} 为数值目标：{len(self.rows)} 行样本中 {len(self.rows) - missing} 行有值，"
                f"均值 {num_text(stats['mean'])}，区间 {num_text(stats['min'])} ~ {num_text(stats['max'])}；"
                f"{missing} 行目标为空，不进入训练。"
            )
        elif categories:
            extras["categories"] = categories
            metrics.append({"label": "取值种类", "value": str(len(categories)), "kind": "stat"})
            top = "、".join(f"{item['value']}({item['count']})" for item in categories[:3])
            summary = (
                f"{self.target} 为离散目标：{len(self.rows) - missing} 行有值，取值为 {top}；"
                f"{missing} 行目标为空，不进入训练。"
            )
        else:
            summary = f"{self.target} 在样本中全部为空，目标口径无法从本轮样本得到验证。"
            status = "missing"
            basis.append({
                "kind": "missing",
                "field": self.target,
                "text": f"快照样本中 {self.target} 全部为空，本轮没有可解释的目标分布。",
            })

        included = {index: not is_blank(row.get(self.target)) for index, row in enumerate(self.rows)}
        for index in self.row_drops:
            included[index] = False

        basis.extend(
            {"kind": item["kind"], "field": self.target, "text": item["text"]}
            for item in self.field_notes.get(self.target, [])
        )
        if status != "missing":
            status = "partial" if missing else "explained"
        card = self._card_base(STEP_TARGET, "target", status, summary)
        card["definitions"] = definitions
        card["metrics"] = metrics
        card["basis"] = basis
        card["rows"] = self._evidence_rows([self.target], included)
        card["extras"] = extras
        return card

    # ------------------------------------------------------------------ 特征取值
    def _select_usable_rows(self) -> tuple[list[int], dict[int, bool]]:
        included: dict[int, bool] = {}
        usable: list[int] = []
        for index, row in enumerate(self.rows):
            if index in self.row_drops or is_blank(row.get(self.target)):
                included[index] = False
                continue
            ok = True
            for name in self.features:
                if name not in self.defs or is_blank(self.effective[index].get(name)):
                    ok = False
                    break
            included[index] = ok
            if ok:
                usable.append(index)
        return usable, included

    def _feature_card(self) -> dict[str, Any]:
        definitions = [self._definition_ref(name) for name in self.features]
        basis: list[dict[str, str]] = []
        if not self.features:
            card = self._card_base(STEP_FEATURE, "feature", "missing", "留空：没有登记特征字段。")
            card["basis"] = [{"kind": "missing", "field": "", "text": "模型配置没有登记特征字段，本轮只能解释目标与缺值。"}]
            return card

        usable, included = self._select_usable_rows()
        self._usable_rows = usable
        numeric_targets = [as_number(self.rows[index].get(self.target)) for index in usable]
        target_numeric = bool(numeric_targets) and all(number is not None for number in numeric_targets)
        labels = binary_labels([self.rows[index].get(self.target) for index in usable], sorted(POSITIVE_WORDS))
        target_binary = bool(labels) and sorted(set(labels)) == [0, 1]
        stats: list[dict[str, Any]] = []
        missing_count = 0
        for name in self.features:
            definition = self.defs.get(name)
            if definition is None:
                missing_count += 1
                basis.append({
                    "kind": "missing",
                    "field": name,
                    "text": (
                        f"特征 {name} 不在数据集 {self.dataset.get('code') or '—'} "
                        f"v{self.dataset.get('version') or '—'} 的定义里，类型与编码方式无法解释，本轮不参与拟合。"
                    ),
                })
                stats.append({
                    "name": name,
                    "defined": False,
                    "type": "",
                    "coverage": "0 / 0",
                    "detail": "留空：无当时字段定义",
                    "correlation": "—",
                })
                continue
            field_type = str(definition.get("type") or "")
            values = [self.effective[index].get(name) for index in usable]
            coverage = f"{sum(1 for value in values if not is_blank(value))} / {len(usable)}"
            detail = ""
            correlation = "—"
            group_means: list[dict[str, Any]] | None = None
            if field_type in {"数值", "布尔"}:
                numbers = [number for number in (as_number(value) for value in values) if number is not None]
                numeric = numeric_stats(numbers)
                detail = (
                    f"非空 {len(numbers)} 行，均值 {num_text(numeric['mean'])}，"
                    f"区间 {num_text(numeric['min'])} ~ {num_text(numeric['max'])}"
                )
                if target_numeric:
                    pairs = [
                        (as_number(self.effective[index].get(name)), as_number(self.rows[index].get(self.target)))
                        for index in usable
                    ]
                    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                    corr = pearson([x for x, _ in pairs], [y for _, y in pairs])
                    correlation = num_text(corr) if corr is not None else "—"
                elif target_binary:
                    pairs = [
                        (as_number(self.effective[index].get(name)), float(label))
                        for index, label in zip(usable, labels)
                    ]
                    pairs = [(x, y) for x, y in pairs if x is not None]
                    corr = pearson([x for x, _ in pairs], [y for _, y in pairs])
                    correlation = num_text(corr) if corr is not None else "—"
            elif field_type == "日期":
                ordinals = [self._date_ordinal(self.effective[index].get(name)) for index in usable]
                ordinals = [number for number in ordinals if number is not None]
                detail = f"可解析日期 {len(ordinals)} / {len(usable)} 行"
                if target_numeric:
                    pairs = [
                        (self._date_ordinal(self.effective[index].get(name)), as_number(self.rows[index].get(self.target)))
                        for index in usable
                    ]
                    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                    corr = pearson([x for x, _ in pairs], [y for _, y in pairs])
                    correlation = num_text(corr) if corr is not None else "—"
            else:
                categories = category_stats(values, limit=6)
                top = "、".join(f"{item['value']}({item['count']})" for item in categories[:3])
                detail = f"非空取值 {top or '—'}"
                if target_numeric:
                    groups: dict[str, list[float]] = {}
                    for index in usable:
                        category = display(self.effective[index].get(name))
                        target_value = as_number(self.rows[index].get(self.target))
                        if target_value is not None:
                            groups.setdefault(category, []).append(target_value)
                    group_means = [
                        {"value": category, "mean": num_text(sum(items) / len(items)), "count": len(items)}
                        for category, items in sorted(groups.items(), key=lambda item: -sum(item[1]) / len(item[1]))
                    ]
            entry = {
                "name": name,
                "defined": True,
                "type": field_type,
                "coverage": coverage,
                "detail": detail,
                "correlation": correlation,
            }
            if group_means:
                entry["group_means"] = group_means
            stats.append(entry)

        defined_count = len(self.features) - missing_count
        status = "missing" if defined_count == 0 else ("partial" if missing_count else "explained")
        summary = (
            f"本轮登记 {len(self.features)} 个特征（{'、'.join(self.features)}），"
            f"其中 {defined_count} 个有当时定义、{missing_count} 个缺失；"
            f"可用训练行 {len(usable)} / {len(self.rows)}。"
        )
        basis.extend(
            {"kind": item["kind"], "field": name, "text": item["text"]}
            for name in self.features
            for item in self.field_notes.get(name, [])
        )
        card = self._card_base(STEP_FEATURE, "feature", status, summary)
        card["definitions"] = definitions
        card["metrics"] = [
            {"label": "特征数", "value": str(len(self.features)), "kind": "stat"},
            {"label": "有定义", "value": str(defined_count), "kind": "stat"},
            {"label": "可用行", "value": f"{len(usable)} / {len(self.rows)}", "kind": "stat"},
        ]
        card["basis"] = basis
        card["rows"] = self._evidence_rows([self.target] + self.features, included)
        card["extras"] = {"features": stats}
        return card

    def _date_ordinal(self, value: Any) -> float | None:
        if is_blank(value):
            return None
        try:
            parsed = date.fromisoformat(str(value)[:10])
        except ValueError:
            return None
        return float(parsed.toordinal())

    # ------------------------------------------------------------------ 缺值处理
    def _missing_card(self) -> dict[str, Any]:
        names = list(dict.fromkeys(([self.target] if self.target else []) + self.features))
        if not names:
            return self._card_base(STEP_MISSING, "missing", "missing", "留空：没有目标或特征字段需要处理。")
        details: list[dict[str, Any]] = []
        basis: list[dict[str, str]] = []
        total_filled = 0
        failed_fields: list[str] = []
        missing_policy_fields: list[str] = []
        for name in names:
            info = self.analysis[name]
            affected = info["raw_missing"]
            fill_value = info["fill_value"]
            if fill_value is not None:
                total_filled += affected
            if info["failed"]:
                failed_fields.append(name)
            if not info["defined"] or not info["policy"]:
                missing_policy_fields.append(name)
            details.append({
                "name": name,
                "defined": info["defined"],
                "type": info["type"],
                "policy": info["policy"] or "—",
                "missing": affected,
                "fill_kind": info["fill_kind"] or "—",
                "fill_value": display(fill_value) if fill_value is not None else "留空",
                "filled": affected if fill_value is not None else 0,
                "failed": bool(info["failed"]),
                "notes": info["notes"],
            })
            for item in info["notes"]:
                if item["kind"] == "missing":
                    basis.append({"kind": "missing", "field": name, "text": item["text"]})

        usable, included = self._select_usable_rows()
        switched = [name for name, info in self.analysis.items() if info["fill_kind"]]
        status = "missing" if (failed_fields or missing_policy_fields) else "explained"
        parts = [
            f"本轮涉及 {len(names)} 个字段，{len(switched)} 个字段执行了填充，共补入 {total_filled} 个空值"
            if total_filled
            else f"本轮涉及 {len(names)} 个字段，没有空值需要填充",
            f"{len(failed_fields)} 个字段的填充策略无法执行" if failed_fields else "",
            f"{len(missing_policy_fields)} 个字段缺少当时定义或缺值策略" if missing_policy_fields else "",
            f"最终可用训练行 {len(usable)} / {len(self.rows)} 行",
        ]
        summary = "；".join(part for part in parts if part) + "。"

        raw_columns = names
        rows = self._evidence_rows(raw_columns, included)
        for item, index in zip(rows["items"], range(len(self.rows))):
            for name in raw_columns:
                raw = self.rows[index].get(name)
                filled = self.effective[index].get(name)
                if is_blank(raw) and filled is not None:
                    item["cells"][name] = f"{display(raw)} → {display(filled)}"
                elif not _same_cell(raw, filled):
                    item["cells"][name] = f"{display(raw)} → {display(filled)}"
        card = self._card_base(STEP_MISSING, "missing", status, summary)
        card["definitions"] = [self._definition_ref(name) for name in raw_columns]
        card["metrics"] = [
            {"label": "涉及字段", "value": str(len(names)), "kind": "stat"},
            {"label": "执行填充", "value": str(len(switched)), "kind": "stat"},
            {"label": "补入空值", "value": str(total_filled), "kind": "stat"},
            {"label": "可用训练行", "value": f"{len(usable)} / {len(self.rows)}", "kind": "stat"},
        ]
        card["basis"] = basis
        card["rows"] = rows
        card["extras"] = {"fields": details}
        return card

    # ------------------------------------------------------------------ 评估指标
    def _metric_card(self) -> dict[str, Any]:
        definitions = [self._definition_ref(self.target)] if self.target else []
        basis: list[dict[str, str]] = []
        metric = self.metric
        if not metric:
            return self._unavailable(
                "模型没有配置评估指标，本轮留空。",
                "模型配置里的评估指标为空，无法确定用哪一个口径验收。",
                definitions,
            )
        if not self.target:
            return self._unavailable(
                "留空：模型没有目标字段，评估无法成立。",
                "没有登记目标字段，任何评估指标都缺少被评估的对象。",
                definitions,
            )
        if self.target_def is None:
            return self._unavailable(
                f"留空：目标字段 {self.target} 没有当时定义，评估口径无法成立。",
                f"数据集 v{self.dataset.get('version') or '—'} 中没有 {self.target}，"
                f"评估指标 {metric} 缺少可解释的目标类型。",
                definitions,
            )
        usable, _ = self._select_usable_rows()
        if not usable:
            return self._unavailable(
                "留空：没有满足条件的训练行，指标无法计算。",
                "目标为空或被缺值策略整行退出后，样本中没有可用训练行。",
                definitions,
            )

        metric_name = metric.upper() if metric.upper() in REGRESSION_METRICS | CLASSIFICATION_METRICS else metric
        target_values = [self.rows[index].get(self.target) for index in usable]
        numeric_targets = [as_number(value) for value in target_values]
        target_numeric = all(number is not None for number in numeric_targets)
        labels = binary_labels(target_values, sorted(POSITIVE_WORDS))
        distinct = sorted(set(labels))
        target_binary = distinct == [0, 1] and len(target_values) > 1
        target_type = str(self.target_def.get("type") or "")

        if metric_name in REGRESSION_METRICS:
            if not target_numeric:
                return self._unavailable(
                    f"留空：{metric} 需要数值目标，当前目标是「{target_type or '未知'}」。",
                    f"目标字段 {self.target} 的取值不是数值，{metric} 无法按回归口径计算；本轮保留指标项为空。",
                    definitions,
                )
            return self._regression_card(metric_name, usable, [number for number in numeric_targets if number is not None], basis, definitions)
        if metric_name in {"AUC", "F1"}:
            if not target_binary:
                return self._unavailable(
                    f"留空：{metric} 需要二分类目标，当前目标不是二分类。",
                    f"目标字段 {self.target} 的可用样本取值无法稳定分成两个类别，{metric} 本轮无法计算。",
                    definitions,
                )
            return self._binary_card(metric_name, usable, labels, basis, definitions)
        if metric_name in {"准确率", "ACC", "ACCURACY"}:
            if target_numeric and len(set(numeric_targets)) > 2:
                return self._unavailable(
                    "留空：准确率需要分类目标，当前数值目标有多个取值。",
                    f"目标字段 {self.target} 是连续数值，准确率无法定义分类正确的含义。",
                    definitions,
                )
            return self._majority_card(usable, target_values, basis, definitions)
        return self._unavailable(
            f"留空：评估指标「{metric}」没有对应的解释口径。",
            f"证据解释器只支持 {', '.join(sorted(REGRESSION_METRICS | CLASSIFICATION_METRICS))}，"
            f"当前配置「{metric}」不在其中。",
            definitions,
        )

    def _regression_card(
        self,
        metric_name: str,
        usable: list[int],
        actual: list[float],
        basis: list[dict[str, str]],
        definitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        encoded, feature_notes, _ = self._encode_features(usable)
        weights = fit_linear(encoded, actual) if encoded else []
        predictions = [predict(weights, row) for row in encoded] if encoded else []
        if not predictions:
            mean = sum(actual) / len(actual)
            predictions = [mean] * len(actual)
            extras_baseline = f"仅均值基线（没有可编码特征），预测 {num_text(mean)}"
        else:
            extras_baseline = ""
        metrics = regression_metrics(actual, predictions)
        metric_value = metrics.get(metric_name)
        baseline_mean = sum(actual) / len(actual)
        baseline_metrics = regression_metrics(actual, [baseline_mean] * len(actual))
        baseline_value = baseline_metrics.get(metric_name)
        improvement = ""
        if metric_value is not None and baseline_value not in (None, 0):
            delta = (baseline_value - metric_value) / baseline_value * 100
            improvement = f"{num_text(delta, 1)}%（相对均值基线 {num_text(baseline_value)}，误差越低越好）"
        elif baseline_value is not None:
            improvement = f"均值基线 {num_text(baseline_value)}，本轮组合特征未带来增益"
        if metric_value is None:
            return self._unavailable(
                f"留空：{metric_name} 无法从本轮样本算出有效值。",
                f"样本中目标分布不足以计算 {metric_name}。",
                definitions,
            )
        if metric_name == "MAPE" and any(abs(number) < 1e-9 for number in actual):
            basis.append({"kind": "note", "field": self.target, "text": "目标值中存在 0，MAPE 会跳过这些行。"})

        rows_values = []
        for index, prediction, actual_value in zip(usable, predictions, actual):
            rows_values.append({
                "row_index": index,
                "cells": {
                    "row_index": index + 1,
                    "actual": display(actual_value),
                    "predicted": display(prediction),
                    "error": display(abs(actual_value - prediction)),
                },
                "included": True,
            })
        summary = (
            f"按 {metric_name} 评估，本轮 {len(usable)} 行可用样本上得 {num_text(metric_value)}；"
            f"均值基线为 {num_text(baseline_value)}。"
        )
        card = self._card_base(STEP_METRIC, "metric", "explained", summary)
        card["definitions"] = definitions
        card["metrics"] = [
            {"label": metric_name, "value": num_text(metric_value, 4), "kind": "metric", "hint": "越低越好"},
            {"label": "均值基线", "value": num_text(baseline_value), "kind": "baseline"},
            {"label": "样本行", "value": str(len(usable)), "kind": "stat"},
        ]
        card["metrics"].extend(
            {"label": label, "value": value, "kind": "metric", "hint": "越低越好"}
            for label, value in (
                ("MAE", metrics.get("MAE")),
                ("RMSE", metrics.get("RMSE")),
                ("MAPE", metrics.get("MAPE")),
            )
            if label != metric_name
        )
        card["basis"] = basis
        card["rows"] = {
            "columns": [
                {"key": "row_index", "label": "样本行", "type": "index"},
                {"key": "actual", "label": "实际值", "type": "number"},
                {"key": "predicted", "label": "拟合值", "type": "number"},
                {"key": "error", "label": "绝对误差", "type": "number"},
            ],
            "items": rows_values,
        }
        card["extras"] = {
            "metric": metric_name,
            "target_type": "数值",
            "usable_rows": len(usable),
            "features_encoded": feature_notes,
            "baseline": extras_baseline or f"均值基线 {num_text(baseline_mean)}",
            "baseline_value": baseline_value,
            "improvement": improvement,
            "weights": weights,
        }
        self._fit_summary = {"metric": metric_name, "value": metric_value, "usable_rows": len(usable)}
        return card

    def _binary_card(
        self,
        metric_name: str,
        usable: list[int],
        labels: list[int],
        basis: list[dict[str, str]],
        definitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        encoded, feature_notes, _ = self._encode_features(usable)
        weights = fit_linear(encoded, [float(label) for label in labels]) if encoded else [0.0]
        scores = [predict(weights, row) for row in encoded]
        metric_value: float | None
        extras_precision = ""
        extras_recall = ""
        if metric_name == "AUC":
            metric_value = auc_score(labels, scores)
        else:
            guessed = [1 if score >= 0.5 else 0 for score in scores]
            tp = sum(1 for guess, label in zip(guessed, labels) if guess == 1 and label == 1)
            fp = sum(1 for guess, label in zip(guessed, labels) if guess == 1 and label == 0)
            fn = sum(1 for guess, label in zip(guessed, labels) if guess == 0 and label == 1)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            if metric_name == "F1":
                metric_value = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            else:
                metric_value = sum(1 for guess, label in zip(guessed, labels) if guess == label) / len(labels)
            extras_precision = num_text(precision)
            extras_recall = num_text(recall)
        if metric_value is None:
            return self._unavailable(
                f"留空：{metric_name} 无法从本轮样本算出有效值。",
                f"样本中只有单一类别的目标，{metric_name} 没有可比的正负类对。",
                definitions,
            )
        minority = max(sum(labels), len(labels) - sum(labels)) / len(labels)
        rows_values = []
        for index, score, label in zip(usable, scores, labels):
            rows_values.append({
                "row_index": index,
                "cells": {
                    "row_index": index + 1,
                    "actual": "正类" if label == 1 else "负类",
                    "score": display(score),
                    "predicted": "正类" if score >= 0.5 else "负类",
                },
                "included": True,
            })
        summary = (
            f"按 {metric_name} 评估，本轮 {len(usable)} 行可用样本上得 {num_text(metric_value)}；"
            f"正类 {sum(labels)} 行、负类 {len(labels) - sum(labels)} 行。"
        )
        card = self._card_base(STEP_METRIC, "metric", "explained", summary)
        card["definitions"] = definitions
        card["metrics"] = [
            {"label": metric_name, "value": num_text(metric_value, 4), "kind": "metric", "hint": "越高越好"},
            {"label": "正类行", "value": str(sum(labels)), "kind": "stat"},
            {"label": "负类行", "value": str(len(labels) - sum(labels)), "kind": "stat"},
            {"label": "多数类占比", "value": num_text(minority), "kind": "stat"},
        ]
        if extras_precision:
            card["metrics"].append({"label": "精确率", "value": extras_precision, "kind": "metric"})
            card["metrics"].append({"label": "召回率", "value": extras_recall, "kind": "metric"})
        card["basis"] = basis
        card["rows"] = {
            "columns": [
                {"key": "row_index", "label": "样本行", "type": "index"},
                {"key": "actual", "label": "实际类别", "type": "badge"},
                {"key": "score", "label": "模型分数", "type": "number"},
                {"key": "predicted", "label": "判定类别", "type": "badge"},
            ],
            "items": rows_values,
        }
        card["extras"] = {
            "metric": metric_name,
            "target_type": "二分类",
            "usable_rows": len(usable),
            "features_encoded": feature_notes,
            "baseline": f"多数类占比 {num_text(minority)}",
            "improvement": "",
            "weights": weights,
            "precision": extras_precision,
            "recall": extras_recall,
        }
        self._fit_summary = {"metric": metric_name, "value": metric_value, "usable_rows": len(usable)}
        return card

    def _majority_card(
        self,
        usable: list[int],
        target_values: list[Any],
        basis: list[dict[str, str]],
        definitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for value in target_values:
            key = display(value)
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return self._unavailable(
                "留空：目标取值全部为空，准确率无法计算。",
                "可用训练行的目标字段没有非空取值。",
                definitions,
            )
        majority = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        hits = counts[majority]
        metric_value = hits / len(target_values)
        rows_values = []
        for index, value in zip(usable, target_values):
            rows_values.append({
                "row_index": index,
                "cells": {
                    "row_index": index + 1,
                    "actual": display(value),
                    "predicted": majority,
                    "correct": "是" if display(value) == majority else "否",
                },
                "included": True,
            })
        summary = (
            f"按准确率评估，本轮 {len(usable)} 行可用样本上多数类判定得 {num_text(metric_value)}；"
            f"多数类为「{majority}」，占 {hits} 行。"
        )
        card = self._card_base(STEP_METRIC, "metric", "explained", summary)
        card["definitions"] = definitions
        card["metrics"] = [
            {"label": "准确率", "value": num_text(metric_value, 4), "kind": "metric", "hint": "越高越好"},
            {"label": "多数类", "value": majority, "kind": "stat"},
            {"label": "命中行", "value": f"{hits} / {len(target_values)}", "kind": "stat"},
        ] + [
            {"label": key, "value": str(value), "kind": "stat"}
            for key, value in sorted(counts.items(), key=lambda item: -item[1])[:3]
        ]
        card["basis"] = basis
        card["rows"] = {
            "columns": [
                {"key": "row_index", "label": "样本行", "type": "index"},
                {"key": "actual", "label": "实际类别", "type": "text"},
                {"key": "predicted", "label": "多数类判定", "type": "text"},
                {"key": "correct", "label": "是否正确", "type": "badge"},
            ],
            "items": rows_values,
        }
        card["extras"] = {
            "metric": "准确率",
            "target_type": "分类",
            "usable_rows": len(usable),
            "features_encoded": [],
            "baseline": f"多数类「{majority}」基线，覆盖 {hits} / {len(target_values)} 行",
            "improvement": "",
            "weights": [],
            "classes": [{"value": key, "count": value} for key, value in sorted(counts.items(), key=lambda item: -item[1])],
        }
        self._fit_summary = {"metric": "准确率", "value": metric_value, "usable_rows": len(usable)}
        return card

    def _encode_features(self, usable: list[int]) -> tuple[list[list[float]], list[dict[str, Any]], dict[str, float]]:
        encoded: list[list[float]] = []
        notes: list[dict[str, Any]] = []
        target_values = [as_number(self.rows[index].get(self.target)) for index in usable]
        if target_values and all(value is not None for value in target_values):
            fallback = sum(value for value in target_values if value is not None) / len(target_values)
        else:
            labels = binary_labels([self.rows[index].get(self.target) for index in usable], sorted(POSITIVE_WORDS))
            fallback = sum(labels) / len(labels) if labels else 0.0
        category_means: dict[str, dict[str, float]] = {}
        for name in self.features:
            definition = self.defs.get(name)
            if definition is None or str(definition.get("type")) != "文本":
                continue
            groups: dict[str, list[float]] = {}
            for index in usable:
                category = display(self.effective[index].get(name))
                value = as_number(self.rows[index].get(self.target))
                if value is None:
                    labels = binary_labels([self.rows[index].get(self.target)], sorted(POSITIVE_WORDS))
                    value = float(labels[0]) if labels else None
                if value is not None:
                    groups.setdefault(category, []).append(value)
            category_means[name] = {
                category: sum(items) / len(items) for category, items in groups.items()
            }
        for index in usable:
            row_vector: list[float] = []
            for name in self.features:
                definition = self.defs.get(name)
                value = self.effective[index].get(name)
                if definition is None:
                    row_vector.append(0.0)
                    continue
                field_type = str(definition.get("type") or "")
                if field_type == "日期":
                    ordinal = self._date_ordinal(value)
                    row_vector.append(0.0 if ordinal is None else ordinal)
                elif field_type == "文本":
                    category = display(value)
                    row_vector.append(category_means.get(name, {}).get(category, fallback))
                elif field_type == "布尔":
                    row_vector.append(1.0 if str(value).strip().lower() in {"1", "true", "是", "开启"} else 0.0)
                else:
                    number = as_number(value)
                    row_vector.append(0.0 if number is None else number)
            encoded.append(row_vector)
        for name in self.features:
            definition = self.defs.get(name)
            if definition is None:
                notes.append({"name": name, "encoding": "留空（无定义）"})
            elif str(definition.get("type")) == "文本":
                notes.append({"name": name, "encoding": "类别目标均值编码"})
            elif str(definition.get("type")) == "日期":
                notes.append({"name": name, "encoding": "日期序数"})
            elif str(definition.get("type")) == "布尔":
                notes.append({"name": name, "encoding": "0/1"})
            else:
                notes.append({"name": name, "encoding": "原值"})
        return encoded, notes, category_means

    def _unavailable(
        self,
        summary: str,
        reason: str,
        definitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        card = self._card_base(STEP_METRIC, "metric", "missing", summary)
        card["definitions"] = definitions
        card["basis"] = [{"kind": "missing", "field": self.target, "text": reason}]
        card["metrics"] = [{"label": self.metric or "评估指标", "value": "留空", "kind": "metric"}]
        card["extras"] = {"metric": self.metric, "unavailable": reason, "target_type": "", "usable_rows": 0}
        self._fit_summary = {"metric": self.metric, "value": None}
        return card
