"""内存数据仓库：给每个业务模块准备一份可筛选、可流转的示例数据。

真实项目里这里会换成数据库访问层；当前实现只依赖标准库，保证克隆下来就能起。
"""
from __future__ import annotations

from __future__ import annotations

import copy
from typing import Any

from app.seed import SEED_ROWS
from app.seed_evidence import SEED_EVIDENCE


class Store:
    def __init__(self) -> None:
        self._tables: dict[str, list[dict[str, Any]]] = {
            name: [copy.deepcopy(row) for row in rows] for name, rows in SEED_ROWS.items()
        }
        # 数据集管理与趋势预测换成带结构化字段定义、样本行和版本链的示例数据：
        # 训练证据解释需要真实可算的输入，占位字符串无法支撑核对。
        # 深层拷贝：修订版本与改样本都会就地追加，不能让种子数据被运行时改坏。
        for name, rows in SEED_EVIDENCE.items():
            self._tables[name] = [copy.deepcopy(row) for row in rows]

    def module_names(self) -> list[str]:
        return sorted(self._tables)

    def rows(self, module: str) -> list[dict[str, Any]]:
        return self._tables.setdefault(module, [])

    def find(self, module: str, entry_id: int) -> dict[str, Any] | None:
        for row in self.rows(module):
            if int(row.get("id", 0)) == entry_id:
                return row
        return None

    def overview(self) -> dict[str, object]:
        modules: list[dict[str, object]] = []
        for name in self.module_names():
            rows = self.rows(name)
            modules.append({
                "name": name,
                "created": len(rows),
                "pending": sum(1 for row in rows if row.get("pending")),
                "abnormal": sum(1 for row in rows if row.get("abnormal")),
            })
        cards = [
            {"label": "业务模块", "value": len(modules)},
            {"label": "今日新增", "value": sum(int(item["created"]) for item in modules)},
            {"label": "待处理", "value": sum(int(item["pending"]) for item in modules)},
            {"label": "异常量", "value": sum(int(item["abnormal"]) for item in modules)},
        ]
        return {"cards": cards, "modules": modules}


store = Store()
