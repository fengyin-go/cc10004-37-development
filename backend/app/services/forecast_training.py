"""训练轮次与签认：冻结输入版本，解释按轮次裁决，签认原子封存。

关键规则：
- 发起训练时在锁内深拷贝数据集定义版本与样本行，形成不可变快照；
- 解释结果按轮次存放，最新轮次占用「当前评估」位置，旧轮解释迟到只进历史；
- 签认只在解释完成后允许一次生效，之后的重复请求幂等返回同一封存结果；
- 已签认的证据卡永远来自封存快照，数据集后来删列、改名都不会改写它。
"""
from __future__ import annotations

import copy
import threading
from datetime import datetime
from typing import Any

from app.evidence import EvidenceEngine
from app.evidence_stats import (
    stable_fingerprint,
)
from app.services.lock import STORE_LOCK
from app.store import store

MODULE = "forecast"
DATA_MODULE = "datastore"

STATUS_PENDING = "解释生成中"
STATUS_READY = "待签认"
STATUS_SEALED = "已签认"
STATUS_LATE = "迟到解释"

POSITIVE_WORDS = {"1", "true", "是", "已退款", "退款", "流失", "转化"}

# 轮次按模型分开存放在进程内；真实项目会落库，这里与内存仓库保持一致的易用性。
_ROUNDS: dict[int, list[dict[str, Any]]] = {}
_ROUND_SEQ: dict[int, int] = {}
_ROUND_ID_SEQ = [0]
_ROUNDS_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _find_dataset(code: str) -> dict[str, Any] | None:
    if not code:
        return None
    for row in store.rows(DATA_MODULE):
        if str(row.get("数据集编号")) == code:
            return row
    return None


def _dataset_version(row: dict[str, Any], version: int | None = None) -> dict[str, Any] | None:
    versions = row.get("定义版本") or []
    if not versions:
        return None
    if version is None:
        return versions[-1]
    for item in versions:
        if int(item.get("version", 0)) == version:
            return item
    return None


def _current_dataset_version(row: dict[str, Any] | None) -> int:
    if not row:
        return 0
    versions = row.get("定义版本") or []
    return int(versions[-1].get("version", 0)) if versions else 0


def parse_features(raw: Any) -> list[str]:
    if not raw:
        return []
    text = str(raw).replace("，", ",").replace("、", ",").replace("；", ",").replace(";", ",")
    return [part.strip() for part in text.split(",") if part.strip()]


def _round_summary(round_: dict[str, Any]) -> dict[str, Any]:
    sealed = round_.get("sealed")
    explanation = (sealed or {}).get("explanation") or round_.get("explanation") or {}
    return {
        "round_id": round_.get("round_id"),
        "seq": round_.get("seq"),
        "status": round_.get("status"),
        "status_text": round_.get("status_text") or round_.get("status"),
        "created_at": round_.get("created_at"),
        "operator": round_.get("operator"),
        "note": round_.get("note") or "",
        "late": bool(round_.get("late")),
        "effective": bool(round_.get("effective")),
        "signed": sealed is not None,
        "sealed_fingerprint": (sealed or {}).get("指纹"),
        "signed_at": (sealed or {}).get("签认时间"),
        "signed_by": (sealed or {}).get("签认人"),
        "metric_values": explanation.get("metric_values") or {},
        "gaps": len(explanation.get("gaps") or []),
        "input_version": round_.get("snapshot", {}).get("dataset", {}).get("version"),
        "input_dataset": round_.get("snapshot", {}).get("dataset", {}).get("code"),
        "sample_rows": len(round_.get("snapshot", {}).get("sample") or []),
    }


class TrainingService:
    def __init__(self) -> None:
        pass

    # ---------------------------------------------------------------- 列表与详情
    def list_rounds(self, entry_id: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        entry = store.find(MODULE, entry_id)
        if entry is None:
            return None, {}
        rounds = self._model_rounds(entry_id)
        ordered = sorted(rounds, key=lambda item: item.get("seq", 0), reverse=True)
        current = self._current_round(entry_id)
        latest_explained = self._latest_explained_round(entry_id)
        payload = {
            "model": entry,
            "dataset_code": str(entry.get("训练数据集") or ""),
            "rounds": [_round_summary(item) for item in ordered],
            "current_round_id": current.get("round_id") if current else None,
            "current_round_seq": current.get("seq") if current else 0,
            "current_round_status": current.get("status") if current else "",
            "displayed_round_id": latest_explained.get("round_id") if latest_explained else None,
            "sealed_round_ids": [item["round_id"] for item in ordered if item.get("sealed")],
            "dataset_version": _current_dataset_version(_find_dataset(str(entry.get("训练数据集") or ""))),
        }
        return entry, payload

    def get_round(self, entry_id: int, round_id: int) -> dict[str, Any] | None:
        self._model_rounds(entry_id)
        for item in _ROUNDS.get(entry_id, []):
            if int(item.get("round_id", 0)) == round_id:
                return self._round_payload(entry_id, item)
        return None

    def current_round(self, entry_id: int) -> dict[str, Any] | None:
        round_ = self._current_round(entry_id)
        return self._round_payload(entry_id, round_) if round_ else None

    def _round_payload(self, entry_id: int, round_: dict[str, Any]) -> dict[str, Any]:
        current = self._current_round(entry_id)
        summary = _round_summary(round_)
        sealed = round_.get("sealed")
        explanation = sealed.get("explanation") if sealed else round_.get("explanation")
        payload: dict[str, Any] = {
            "summary": summary,
            "snapshot": round_.get("snapshot"),
            "explanation": explanation,
            "sealed": sealed,
            "is_current": bool(current and current.get("round_id") == round_.get("round_id")),
            "late": bool(round_.get("late")),
            "drift": [],
            "superseded_by": round_.get("superseded_by"),
        }
        if explanation is None:
            payload["message"] = (
                f"第 {round_.get('seq')} 轮的解释尚未生成；当前评估位由第 "
                f"{current.get('seq') if current else '—'} 轮占用。"
            )
        if sealed:
            payload["message"] = "该轮已签认，以下内容为签认时原子封存的原样，不受数据集后续变更影响。"
        elif round_.get("late"):
            payload["message"] = (
                f"第 {round_.get('seq')} 轮解释迟到：第 {round_.get('superseded_by')} 轮已占据当前评估，"
                "本轮只保留在历史中，不覆盖新评估。"
            )
        payload["drift"] = self._drift(round_)
        return payload

    def _drift(self, round_: dict[str, Any]) -> list[dict[str, Any]]:
        snapshot_dataset = (round_.get("snapshot") or {}).get("dataset") or {}
        snapshot_fields = [str(item.get("name")) for item in snapshot_dataset.get("fields") or []]
        dataset = _find_dataset(str(snapshot_dataset.get("code") or ""))
        if dataset is None:
            return [{
                "kind": "dataset_missing",
                "text": f"数据集 {snapshot_dataset.get('code') or '—'} 已不在数据集管理里；签认卡仍按第 {round_.get('seq')} 轮快照展示。",
            }]
        current = _dataset_version(dataset)
        current_fields = [str(item.get("name")) for item in (current or {}).get("fields") or []]
        current_version = int((current or {}).get("version", 0))
        notes: list[dict[str, Any]] = []
        if current_version != int(snapshot_dataset.get("version") or 0):
            notes.append({
                "kind": "version_changed",
                "text": f"数据集已从 v{snapshot_dataset.get('version')} 修订到 v{current_version}；本轮卡片仍引用 v{snapshot_dataset.get('version')}。",
            })
        removed = [name for name in snapshot_fields if name not in current_fields]
        added = [name for name in current_fields if name not in snapshot_fields]
        if removed:
            notes.append({"kind": "field_removed", "text": f"快照字段 {('、'.join(removed))} 在当前定义里已不存在。"})
        if added:
            notes.append({"kind": "field_added", "text": f"当前定义新增了 {('、'.join(added))}，不影响本轮快照。"})
        return notes

    # ---------------------------------------------------------------- 发起训练
    def start_round(
        self,
        entry_id: int,
        *,
        note: str = "",
        operator: str = "值班管理员",
        delay_ms: int = 0,
        defer: bool = False,
    ) -> tuple[dict[str, Any] | None, str]:
        with STORE_LOCK:
            entry = store.find(MODULE, entry_id)
            if entry is None:
                return None, f"预测模型 {entry_id} 不存在或已归档"
            dataset_code = str(entry.get("训练数据集") or "").strip()
            if not dataset_code:
                return None, "预测模型没有登记训练数据集，无法开启训练轮次"
            snapshot = self._build_snapshot(entry, dataset_code)
            if snapshot is None:
                return None, f"训练数据集 {dataset_code} 不存在，无法开启训练轮次"
            _ROUND_SEQ[entry_id] = _ROUND_SEQ.get(entry_id, 0) + 1
            with _ROUNDS_LOCK:
                _ROUND_ID_SEQ[0] += 1
                round_id = _ROUND_ID_SEQ[0]
                round_ = {
                    "round_id": round_id,
                    "model_id": entry_id,
                    "seq": _ROUND_SEQ[entry_id],
                    "status": STATUS_PENDING,
                    "status_text": "解释生成中",
                    "created_at": _now(),
                    "operator": operator,
                    "note": note,
                    "snapshot": snapshot,
                    "explanation": None,
                    "sealed": None,
                    "late": False,
                    "effective": False,
                    "superseded_by": None,
                    "explained_at": None,
                }
                _ROUNDS.setdefault(entry_id, []).append(round_)
                self._refresh_statuses(entry_id)
        # 快照已冻结，解释计算不再读取当前定义，所以可以安全地在锁外慢慢算。
        if defer:
            return self._round_payload(entry_id, round_), f"第 {round_['seq']} 轮训练已发起，输入版本已冻结，等待解释生成"
        if delay_ms > 0:
            timer = threading.Thread(
                target=self._delayed_explain,
                args=(entry_id, round_id, min(delay_ms, 5000) / 1000.0),
                daemon=True,
            )
            timer.start()
            return self._round_payload(entry_id, round_), f"第 {round_['seq']} 轮训练已发起，解释将在 {delay_ms} 毫秒后到达"
        return self.explain(entry_id, round_id, operator=operator)

    def _delayed_explain(self, entry_id: int, round_id: int, seconds: float) -> None:
        if seconds > 0:
            threading.Event().wait(seconds)
        try:
            self.explain(entry_id, round_id, operator="解释器")
        except Exception:  # noqa: BLE001 - 后台线程不能把进程带崩
            pass

    def _build_snapshot(self, entry: dict[str, Any], dataset_code: str) -> dict[str, Any] | None:
        dataset = _find_dataset(dataset_code)
        if dataset is None:
            return None
        version = _dataset_version(dataset)
        if version is None:
            return None
        features = parse_features(entry.get("特征字段"))
        snapshot = {
            "dataset": {
                "id": dataset.get("id"),
                "code": dataset_code,
                "name": dataset.get("数据集名称"),
                "source_table": dataset.get("来源表名"),
                "status": dataset.get("status"),
                "version": int(version.get("version", 0)),
                "version_created_at": version.get("created_at"),
                "version_note": version.get("note"),
                "fields": copy.deepcopy(version.get("fields") or []),
            },
            "model": {
                "id": entry.get("id"),
                "code": entry.get("模型编号"),
                "type": entry.get("模型类型"),
                "target": str(entry.get("目标字段") or "").strip(),
                "features": features,
                "metric": str(entry.get("评估指标") or "").strip(),
                "horizon": entry.get("预测步长"),
                "confidence": entry.get("置信区间"),
            },
            "sample": copy.deepcopy(dataset.get("样本数据") or []),
            "sample_note": dataset.get("样本说明") or "",
            "captured_at": _now(),
        }
        snapshot["sample_fingerprint"] = stable_fingerprint({
            "version": snapshot["dataset"]["version"],
            "fields": snapshot["dataset"]["fields"],
            "sample": snapshot["sample"],
            "model": snapshot["model"],
        })
        snapshot["input_fingerprint"] = stable_fingerprint({
            "dataset": snapshot["dataset"],
            "model": snapshot["model"],
            "sample": snapshot["sample"],
        })
        return snapshot

    # ---------------------------------------------------------------- 解释
    def explain(self, entry_id: int, round_id: int, *, operator: str = "解释器") -> tuple[dict[str, Any] | None, str]:
        with STORE_LOCK:
            round_ = self._find_round(entry_id, round_id)
            if round_ is None:
                return None, f"训练轮次 {round_id} 不存在"
            if round_.get("sealed"):
                return self._round_payload(entry_id, round_), "该轮已签认，解释按封存结果幂等返回"
            if round_.get("explanation") is not None:
                return self._round_payload(entry_id, round_), "该轮解释已生成，本次请求幂等返回原结果"
            snapshot = copy.deepcopy(round_.get("snapshot") or {})
        # 锁外计算，计算只依赖快照。
        engine = EvidenceEngine(snapshot)
        explanation = engine.build()
        with STORE_LOCK:
            round_ = self._find_round(entry_id, round_id)
            if round_ is None:
                return None, f"训练轮次 {round_id} 不存在"
            if round_.get("explanation") is not None:
                return self._round_payload(entry_id, round_), "该轮解释已由其他请求生成，本次请求幂等返回原结果"
            latest_seq = max(self._current_round_seqs(entry_id) or [0])
            # 解释到达时就裁定：晚于其他轮次发起的解释不再占用当前评估。
            round_["late"] = round_.get("seq", 0) < latest_seq
            round_["explanation"] = explanation
            round_["explained_at"] = _now()
            round_["effective"] = not round_["late"]
            if round_["late"]:
                superseding = [
                    item for item in _ROUNDS.get(entry_id, [])
                    if item.get("seq", 0) > round_.get("seq", 0)
                ]
                round_["superseded_by"] = min(superseding, key=lambda item: item.get("seq", 0)).get("seq")
            self._refresh_statuses(entry_id)
            message = (
                f"第 {round_.get('seq')} 轮解释迟到：第 {round_.get('superseded_by')} 轮已占据当前评估，"
                "本轮只保留在历史中，不覆盖新评估。"
                if round_.get("late")
                else f"第 {round_.get('seq')} 轮解释已生成，可在核对样本后签认"
            )
            return self._round_payload(entry_id, round_), message

    # ---------------------------------------------------------------- 签认
    def sign_off(
        self,
        entry_id: int,
        round_id: int,
        *,
        operator: str = "值班管理员",
        note: str = "",
    ) -> tuple[dict[str, Any] | None, str]:
        with STORE_LOCK:
            round_ = self._find_round(entry_id, round_id)
            if round_ is None:
                return None, f"训练轮次 {round_id} 不存在"
            if round_.get("sealed") is not None:
                return self._round_payload(entry_id, round_), (
                    f"第 {round_.get('seq')} 轮已签认，本次请求幂等返回封存结果（指纹 "
                    f"{round_['sealed'].get('指纹')}）"
                )
            if round_.get("explanation") is None:
                return None, (
                    f"第 {round_.get('seq')} 轮解释尚未生成，不能签认；签认必须同时封存输入版本与解释。"
                )
            snapshot = copy.deepcopy(round_.get("snapshot") or {})
            explanation = copy.deepcopy(round_.get("explanation") or {})
            sealed = {
                "轮次": round_.get("seq"),
                "轮次编号": round_id,
                "签认时间": _now(),
                "签认人": operator,
                "签认说明": note,
                "输入版本": copy.deepcopy(snapshot.get("dataset") or {}),
                "模型配置": copy.deepcopy(snapshot.get("model") or {}),
                "样本行数": len(snapshot.get("sample") or []),
                "样本": copy.deepcopy(snapshot.get("sample") or []),
                "样本指纹": snapshot.get("sample_fingerprint"),
                "输入指纹": snapshot.get("input_fingerprint"),
                "解释": explanation,
                "迟到": bool(round_.get("late")),
            }
            sealed["解释指纹"] = stable_fingerprint(explanation)
            sealed["指纹"] = stable_fingerprint({
                "输入指纹": sealed["输入指纹"],
                "解释指纹": sealed["解释指纹"],
                "轮次": sealed["轮次"],
            })
            round_["sealed"] = sealed
            round_["status"] = STATUS_SEALED
            round_["status_text"] = "已签认"
            round_["signed_at"] = sealed["签认时间"]
            self._refresh_statuses(entry_id)
            return self._round_payload(entry_id, round_), (
                f"第 {round_.get('seq')} 轮已签认：输入版本 {snapshot.get('dataset', {}).get('code')} "
                f"v{snapshot.get('dataset', {}).get('version')} 与解释已原子封存（指纹 {sealed['指纹']}）"
            )

    # ---------------------------------------------------------------- 内部
    def _model_rounds(self, entry_id: int) -> list[dict[str, Any]]:
        return _ROUNDS.setdefault(entry_id, [])

    def _find_round(self, entry_id: int, round_id: int) -> dict[str, Any] | None:
        for item in self._model_rounds(entry_id):
            if int(item.get("round_id", 0)) == round_id:
                return item
        return None

    def _current_round_seqs(self, entry_id: int) -> list[int]:
        return [int(item.get("seq", 0)) for item in self._model_rounds(entry_id)]

    def _current_round(self, entry_id: int) -> dict[str, Any] | None:
        """当前评估位由序号最大的已生成解释轮次占用；解释未到不提前挤掉现行结果。"""
        rounds = self._model_rounds(entry_id)
        if not rounds:
            return None
        explained = [item for item in rounds if item.get("explanation") is not None]
        pool = explained or rounds
        return max(pool, key=lambda item: (item.get("seq", 0), item.get("round_id", 0)))

    def _latest_explained_round(self, entry_id: int) -> dict[str, Any] | None:
        ready = [
            item for item in self._model_rounds(entry_id)
            if item.get("explanation") is not None
        ]
        if not ready:
            return None
        return max(ready, key=lambda item: (item.get("seq", 0), item.get("round_id", 0)))

    def _refresh_statuses(self, entry_id: int) -> None:
        latest_seq = max(self._current_round_seqs(entry_id) or [0])
        # 尚未生成解释的轮次不参与迟到判定；解释到达时才按发起序号裁决。
        latest_explained_seq = max(
            [item.get("seq", 0) for item in self._model_rounds(entry_id) if item.get("explanation") is not None] or [0]
        )
        for item in self._model_rounds(entry_id):
            if item.get("sealed") is not None:
                item["status"] = STATUS_SEALED
                item["status_text"] = "已签认"
                item["effective"] = False
                continue
            if item.get("explanation") is None:
                item["status"] = STATUS_PENDING
                item["status_text"] = "解释生成中"
                item["effective"] = False
                item["late"] = False
                continue
            if item.get("seq", 0) < latest_seq:
                item["late"] = True
                item["effective"] = False
                item["status"] = STATUS_LATE
                item["status_text"] = "迟到解释"
            else:
                item["late"] = False
                item["effective"] = item.get("seq", 0) == latest_explained_seq
                item["status"] = STATUS_READY
                item["status_text"] = "待签认"


training_service = TrainingService()
