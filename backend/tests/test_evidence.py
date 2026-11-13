"""训练证据解释引擎验收脚本。

用标准库直接调用业务层，不依赖外部 HTTP 客户端或机器学习服务：
    python3 -m unittest discover -s tests -v   （在 backend 目录下、用 .venv 解释器）

覆盖的重点：
1. 改样本会产生不同的解释与指标，不是固定说明文；
2. 签认后封存输入版本与解释，删列 / 改名 / 改缺值策略都不倒写；
3. 缺失环节明确留空并给出无法解释依据，不借当前定义补造历史；
4. 并发轮次按发起顺序裁决，迟到解释不覆盖新评估；
5. 同轮重复签认幂等，返回同一指纹。
"""
from __future__ import annotations

import copy
import sys
import threading
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.services.datastore import DatastoreService  # noqa: E402
from app.services.forecast_training import training_service  # noqa: E402
from app.store import store  # noqa: E402


def _reset_store() -> None:
    """每个用例从种子数据重新开始，避免轮次与修订串场。"""
    from app.seed import SEED_ROWS
    from app.seed_evidence import SEED_EVIDENCE

    for name, rows in SEED_ROWS.items():
        store._tables[name] = [copy.deepcopy(row) for row in rows]
    for name, rows in SEED_EVIDENCE.items():
        store._tables[name] = [copy.deepcopy(row) for row in rows]
    training_service._model_rounds(1).clear()
    training_service._model_rounds(2).clear()
    training_service._model_rounds(3).clear()


def _cards(explanation: dict) -> dict[str, dict]:
    return {card["key"]: card for card in explanation["cards"]}


class EvidenceEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        _reset_store()

    def test_sample_change_changes_explanation(self) -> None:
        """修改可录入的小样本必须产生真实差异。"""
        datastore = DatastoreService()
        first, _ = training_service.start_round(1, note="首轮")
        first_metrics = first["explanation"]["metric_values"]
        first_summary = _cards(first["explanation"])["metric"]["summary"]

        rows = copy.deepcopy(store.find("datastore", 1)["样本数据"])
        rows[0]["amount"] = 1000
        rows[2]["amount"] = 900
        datastore.update_sample(1, {"rows": rows, "note": "调大两笔订单"})
        second, _ = training_service.start_round(1, note="改样本后")
        second_metrics = second["explanation"]["metric_values"]
        second_summary = _cards(second["explanation"])["metric"]["summary"]

        self.assertNotEqual(first_metrics["MAE"], second_metrics["MAE"])
        self.assertNotEqual(first_summary, second_summary)
        self.assertIn("1000", _cards(second["explanation"])["target"]["summary"])

    def test_metric_missing_without_definition_after_drop_column(self) -> None:
        """删列后新轮次不能借用当前定义补造，评估指标明确留空。"""
        datastore = DatastoreService()
        datastore.revise(1, {"action": "删除字段", "name": "amount", "note": "删掉金额"})
        round_, _ = training_service.start_round(1, note="删列后")
        cards = _cards(round_["explanation"])
        self.assertEqual(cards["target"]["status"], "missing")
        self.assertEqual(cards["metric"]["status"], "missing")
        self.assertIn("留空", cards["metric"]["summary"])
        gaps = round_["explanation"]["gaps"]
        self.assertTrue(any(gap["card"] == "target" and "不用今天的数据集定义补造" in gap["reason"] for gap in gaps))

    def test_defined_field_but_metric_type_mismatch_is_blank(self) -> None:
        """评估口径与目标类型不匹配时也留空，并给出依据。"""
        store.find("forecast", 1)["评估指标"] = "AUC"
        round_, _ = training_service.start_round(1, note="口径不匹配")
        metric_card = _cards(round_["explanation"])["metric"]
        self.assertEqual(metric_card["status"], "missing")
        self.assertIn("二分类", metric_card["summary"])

    def test_missing_policy_field_is_explained_from_snapshot(self) -> None:
        """缺值处理卡片按快照里的策略计算填充值与受影响行数。"""
        round_, _ = training_service.start_round(1, note="缺值核对")
        missing_card = _cards(round_["explanation"])["missing"]
        self.assertTrue(any("region" in item["name"] and item["policy"] == "众数填充" for item in missing_card["extras"]["fields"]))
        self.assertGreaterEqual(int(missing_card["metrics"][2]["value"]), 1)

    def test_cards_are_row_checkable(self) -> None:
        """每张卡都带可展开的逐行样本；指标卡只展示参与训练的行，其余展示全部快照行。"""
        round_, _ = training_service.start_round(1, note="逐行核对")
        total_rows = len(round_["snapshot"]["sample"])
        for card in round_["explanation"]["cards"]:
            self.assertTrue(card["rows"]["columns"])
            self.assertGreater(len(card["rows"]["items"]), 0)
            if card["key"] == "metric":
                self.assertLessEqual(len(card["rows"]["items"]), total_rows)
                self.assertTrue(all(item["included"] for item in card["rows"]["items"]))
            else:
                self.assertEqual(len(card["rows"]["items"]), total_rows)


class SealingTest(unittest.TestCase):
    def setUp(self) -> None:
        _reset_store()

    def test_signed_round_immune_to_later_rename_and_delete(self) -> None:
        """签认原子封存输入版本与解释，后续删列 / 改名不倒写。"""
        datastore = DatastoreService()
        round_, _ = training_service.start_round(1, note="要签认")
        round_id = round_["summary"]["round_id"]
        signed, _ = training_service.sign_off(1, round_id, operator="审核人", note="通过")
        sealed = signed["sealed"]
        sealed_cards_text = str(sealed["解释"])
        fingerprint = sealed["指纹"]

        datastore.revise(1, {"action": "重命名字段", "name": "amount", "new_name": "pay_amount"})
        datastore.revise(1, {"action": "删除字段", "name": "region"})

        after = training_service.get_round(1, round_id)
        self.assertEqual(after["sealed"]["指纹"], fingerprint)
        self.assertEqual(str(after["sealed"]["解释"]), sealed_cards_text)
        self.assertEqual(_cards(after["sealed"]["解释"])["metric"]["summary"], _cards(round_["explanation"])["metric"]["summary"])
        self.assertTrue(after["drift"])
        self.assertTrue(any(note["kind"] == "field_removed" for note in after["drift"]))
        self.assertEqual(after["summary"]["metric_values"], round_["summary"]["metric_values"])
        self.assertEqual(
            _cards(after["sealed"]["解释"])["metric"]["summary"],
            _cards(round_["explanation"])["metric"]["summary"],
        )

    def test_sign_before_explain_rejected(self) -> None:
        """解释未生成不能签认：签认必须同时封存输入版本与解释。"""
        round_, _ = training_service.start_round(1, note="先冻结", defer=True)
        signed, message = training_service.sign_off(1, round_["summary"]["round_id"])
        self.assertIsNone(signed)
        self.assertIn("不能签认", message)

    def test_repeated_sign_is_idempotent(self) -> None:
        """同轮重复签认幂等，指纹不变。"""
        round_, _ = training_service.start_round(1, note="幂等")
        round_id = round_["summary"]["round_id"]
        first, _ = training_service.sign_off(1, round_id)
        second, message = training_service.sign_off(1, round_id)
        self.assertEqual(first["sealed"]["指纹"], second["sealed"]["指纹"])
        self.assertIn("幂等", message)


class ConcurrencyTest(unittest.TestCase):
    def setUp(self) -> None:
        _reset_store()

    def test_late_explanation_does_not_override_newer_evaluation(self) -> None:
        """迟到解释只进历史，不覆盖新评估。"""
        first, _ = training_service.start_round(1, note="先发起但慢", defer=True)
        second, _ = training_service.start_round(1, note="后发起但快")
        late_first, message = training_service.explain(1, first["summary"]["round_id"])
        self.assertTrue(late_first["late"])
        self.assertEqual(late_first["superseded_by"], second["summary"]["seq"])
        self.assertFalse(late_first["is_current"])
        current = training_service.current_round(1)
        self.assertTrue(current["is_current"])
        self.assertEqual(current["summary"]["seq"], second["summary"]["seq"])
        self.assertIn("不覆盖新评估", message)
        self.assertEqual(store.find("forecast", 1)["id"], 1)

    def test_concurrent_rounds_are_decided_by_sequence(self) -> None:
        """并发发起多轮时按轮次裁决，每组解释结果互不覆盖。"""
        results: list[dict] = []
        errors: list[str] = []

        def run(note: str) -> None:
            try:
                payload, _ = training_service.start_round(1, note=note)
                results.append(payload)
            except Exception as error:  # noqa: BLE001
                errors.append(str(error))

        threads = [threading.Thread(target=run, args=(f"并发{index}",)) for index in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertFalse(errors)
        self.assertEqual(len(results), 5)
        seqs = {item["summary"]["seq"] for item in results}
        self.assertEqual(len(seqs), 5, "每个并发轮次应拿到唯一序号")
        current = training_service.current_round(1)
        self.assertEqual(current["summary"]["seq"], max(seqs))

    def test_pending_round_does_not_evict_current_evaluation(self) -> None:
        """新轮次只冻结了快照、解释未到时，当前评估位仍归上一份已生成的解释。"""
        first, _ = training_service.start_round(1, note="已有解释")
        pending, _ = training_service.start_round(1, note="只冻结", defer=True)
        current = training_service.current_round(1)
        self.assertEqual(current["summary"]["seq"], first["summary"]["seq"])
        self.assertTrue(current["is_current"])
        pending_detail = training_service.get_round(1, pending["summary"]["round_id"])
        self.assertIsNone(pending_detail["explanation"])
        self.assertFalse(pending_detail["is_current"])
        # 解释到达后按发起序号接管当前评估。
        reached, _ = training_service.explain(1, pending["summary"]["round_id"])
        self.assertTrue(reached["is_current"])
        self.assertFalse(reached["late"])

    def test_explanation_uses_frozen_snapshot_not_current_definition(self) -> None:
        """发起时冻结快照：之后改样本、改定义都不影响该轮解释。"""
        datastore = DatastoreService()
        round_, _ = training_service.start_round(1, note="冻结后改样本", defer=True)
        snapshot = copy.deepcopy(round_["snapshot"])
        rows = copy.deepcopy(store.find("datastore", 1)["样本数据"])
        rows[0]["amount"] = 5000
        datastore.update_sample(1, {"rows": rows})
        explained, _ = training_service.explain(1, round_["summary"]["round_id"])
        self.assertEqual(explained["snapshot"], snapshot)
        self.assertNotIn("5000", _cards(explained["explanation"])["target"]["summary"])


class DatastoreRevisionTest(unittest.TestCase):
    def setUp(self) -> None:
        _reset_store()

    def test_versions_keep_history(self) -> None:
        """字段修订按版本追加，历史版本仍可取回。"""
        datastore = DatastoreService()
        datastore.revise(1, {"action": "新增字段", "name": "coupon", "type": "数值", "missing_policy": "零值填充"})
        versions = datastore.list_versions(1)
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0]["version"], 1)
        self.assertNotIn("coupon", [field["name"] for field in versions[0]["fields"]])
        self.assertIn("coupon", [field["name"] for field in versions[1]["fields"]])
        self.assertEqual(datastore.get_version(1, 1)["fields"], versions[0]["fields"])

    def test_sample_rejects_unknown_fields(self) -> None:
        """样本只能写当前定义里的字段，不制造无法解释的列。"""
        datastore = DatastoreService()
        entry, message = datastore.update_sample(1, {"rows": [{"order_id": "X1", "ghost": 1}]})
        self.assertIsNone(entry)
        self.assertIn("ghost", message)


if __name__ == "__main__":
    unittest.main()
