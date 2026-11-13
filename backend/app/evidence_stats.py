"""证据解释器的计算底座：纯 Python 的小样本统计、回归与分类。

这里不依赖任何数值库或外部训练服务；所有指标都由调用方给出的快照样本当场算出，
同一份快照重复计算得到同样结果（用于签认幂等与轮次裁决）。
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def num_text(value: Any, digits: int = 3) -> str:
    number = as_number(value)
    if number is None:
        return "—"
    if number == int(number):
        return str(int(number))
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def display(value: Any) -> str:
    if is_blank(value):
        return "—"
    if isinstance(value, float):
        return num_text(value, 2)
    return str(value)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def numeric_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "median": quantile(values, 0.5),
        "min": min(values),
        "max": max(values),
        "std": math.sqrt(variance),
    }


def category_stats(values: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        if is_blank(value):
            continue
        key = display(value)
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"value": key, "count": count} for key, count in ordered[:limit]]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def stable_fingerprint(payload: Any) -> str:
    """对快照与解释做规范化摘要；不含时间戳，重复计算必须相同。"""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """高斯消元求解线性方程组；奇异时返回 None 交给上层回退到基线。"""
    size = len(rhs)
    rows = [list(matrix[index]) + [rhs[index]] for index in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(rows[row][col]))
        if abs(rows[pivot][col]) < 1e-9:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        factor = rows[col][col]
        rows[col] = [value / factor for value in rows[col]]
        for row in range(size):
            if row == col:
                continue
            ratio = rows[row][col]
            if ratio == 0:
                continue
            rows[row] = [value - ratio * base for value, base in zip(rows[row], rows[col])]
    return [rows[index][size] for index in range(size)]


def fit_linear(features: list[list[float]], targets: list[float], ridge: float = 1e-6) -> list[float]:
    """带截距的最小二乘；设计矩阵含常数列，rank 不足时回退成均值基线。"""
    size = len(features[0]) + 1 if features else 1
    design = [[1.0] + list(row) for row in features]
    normal = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    for row, target in zip(design, targets):
        for i in range(size):
            rhs[i] += row[i] * target
            for j in range(size):
                normal[i][j] += row[i] * row[j]
    for i in range(1, size):
        normal[i][i] += ridge
    weights = _solve(normal, rhs)
    if weights is None:
        mean = sum(targets) / len(targets) if targets else 0.0
        return [mean] + [0.0] * (size - 1)
    return weights


def predict(weights: list[float], feature_row: list[float]) -> float:
    return weights[0] + sum(weight * value for weight, value in zip(weights[1:], feature_row))


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float | None]:
    if not actual:
        return {"MAE": None, "RMSE": None, "MAPE": None}
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    squared = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    mape_items = [abs(a - p) / abs(a) for a, p in zip(actual, predicted) if abs(a) > 1e-9]
    return {
        "MAE": sum(errors) / len(errors),
        "RMSE": math.sqrt(sum(squared) / len(squared)),
        "MAPE": (sum(mape_items) / len(mape_items)) if mape_items else None,
    }


def auc_score(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def binary_labels(values: list[Any], positives: list[str] | None = None) -> list[int]:
    """把布尔/文本目标统一成 0/1；未列出的非空取值都当作正类。"""
    positives = positives or ["1", "true", "是", "已退款", "退款", "流失"]
    labels: list[int] = []
    for value in values:
        if isinstance(value, bool):
            labels.append(1 if value else 0)
        else:
            text = display(value).strip().lower()
            labels.append(1 if text in positives else 0)
    return labels
