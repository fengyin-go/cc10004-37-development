"""训练证据解释所需的示例数据。

数据集管理里的定义不能只是一行占位文本：趋势预测的每一轮训练都要把「当时」的
字段定义、样本行一起快照下来，后面删列、改名都不能倒写已经签认的解释卡。
这里给数据集管理补上结构化的字段版本与小样本，给趋势预测补上可解释的训练配置。
"""
from __future__ import annotations

from typing import Any

DATASET_ORDER = {"数据集编号": "DATA-0001", "数据集名称": "订单交易明细", "来源表名": "dws_order_trade"}


def _field(name: str, type_: str, nullable: bool, policy: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": type_,
        "nullable": nullable,
        "missing_policy": policy,
        "description": description,
    }


def _row(**values: Any) -> dict[str, Any]:
    return values


DATASET_1_FIELDS = [
    _field("order_id", "文本", False, "丢弃该行", "订单唯一编号"),
    _field("pay_time", "日期", False, "丢弃该行", "支付完成时间"),
    _field("region", "文本", True, "众数填充", "订单所属大区"),
    _field("channel", "文本", True, "众数填充", "下单渠道"),
    _field("amount", "数值", True, "均值填充", "订单实付金额（元）"),
    _field("is_refund", "布尔", True, "零值填充", "是否退款"),
]

DATASET_1_SAMPLE = [
    _row(order_id="O1001", pay_time="2026-09-01", region="华东", channel="线上", amount=128.5, is_refund=0),
    _row(order_id="O1002", pay_time="2026-09-02", region="华南", channel="门店", amount=86.0, is_refund=0),
    _row(order_id="O1003", pay_time="2026-09-03", region="华东", channel="线上", amount=212.0, is_refund=0),
    _row(order_id="O1004", pay_time="2026-09-04", region="华北", channel="门店", amount=54.5, is_refund=1),
    _row(order_id="O1005", pay_time="2026-09-05", region=None, channel="线上", amount=301.2, is_refund=0),
    _row(order_id="O1006", pay_time="2026-09-06", region="华南", channel="门店", amount=None, is_refund=0),
    _row(order_id="O1007", pay_time="2026-09-07", region="华东", channel="门店", amount=176.8, is_refund=0),
    _row(order_id="O1008", pay_time="2026-09-08", region="华北", channel="线上", amount=95.0, is_refund=0),
]

DATASET_2_FIELDS = [
    _field("store_id", "文本", False, "丢弃该行", "门店编号"),
    _field("biz_date", "日期", False, "丢弃该行", "营业日期"),
    _field("weekday", "文本", True, "众数填充", "星期几"),
    _field("region", "文本", True, "众数填充", "所属大区"),
    _field("sales_amt", "数值", True, "中位数填充", "当日销售额（万元）"),
    _field("traffic", "数值", True, "均值填充", "进店客流"),
]

DATASET_2_SAMPLE = [
    _row(store_id="S001", biz_date="2026-09-01", weekday="周一", region="华东", sales_amt=12.4, traffic=820),
    _row(store_id="S001", biz_date="2026-09-02", weekday="周二", region="华东", sales_amt=13.1, traffic=865),
    _row(store_id="S001", biz_date="2026-09-03", weekday="周三", region="华东", sales_amt=16.8, traffic=910),
    _row(store_id="S002", biz_date="2026-09-01", weekday="周一", region="华南", sales_amt=9.6, traffic=640),
    _row(store_id="S002", biz_date="2026-09-02", weekday="周二", region="华南", sales_amt=10.2, traffic=702),
    _row(store_id="S002", biz_date="2026-09-03", weekday="周三", region="华南", sales_amt=None, traffic=688),
    _row(store_id="S003", biz_date="2026-09-01", weekday="周一", region="华北", sales_amt=18.3, traffic=1104),
]

DATASET_3_FIELDS = [
    _field("member_id", "文本", False, "丢弃该行", "会员编号"),
    _field("last_active_days", "数值", True, "中位数填充", "距最近活跃天数"),
    _field("login_count", "数值", True, "零值填充", "近 30 天登录次数"),
    _field("tier", "文本", True, "众数填充", "会员等级"),
]

DATASET_3_SAMPLE = [
    _row(member_id="M001", last_active_days=3, login_count=12, tier="金卡"),
    _row(member_id="M002", last_active_days=18, login_count=2, tier="银卡"),
    _row(member_id="M003", last_active_days=1, login_count=21, tier="金卡"),
    _row(member_id="M004", last_active_days=42, login_count=1, tier="普通"),
    _row(member_id="M005", last_active_days=None, login_count=8, tier="银卡"),
    _row(member_id="M006", last_active_days=7, login_count=14, tier="金卡"),
]

DATASET_4_FIELDS = [
    _field("event_id", "文本", False, "丢弃该行", "活动编号"),
    _field("signup_date", "日期", False, "丢弃该行", "报名日期"),
    _field("channel", "文本", True, "众数填充", "投放渠道"),
    _field("headcount", "数值", True, "均值填充", "报名人数"),
    _field("conversion", "数值", True, "均值填充", "到场转化率"),
]

DATASET_4_SAMPLE = [
    _row(event_id="E001", signup_date="2026-09-01", channel="短信", headcount=120, conversion=0.31),
    _row(event_id="E002", signup_date="2026-09-02", channel="社群", headcount=88, conversion=0.46),
    _row(event_id="E003", signup_date="2026-09-03", channel="短信", headcount=150, conversion=0.27),
    _row(event_id="E004", signup_date="2026-09-04", channel="公众号", headcount=65, conversion=0.52),
    _row(event_id="E005", signup_date="2026-09-05", channel="社群", headcount=102, conversion=0.41),
    _row(event_id="E006", signup_date="2026-09-06", channel="公众号", headcount=71, conversion=0.49),
    _row(event_id="E007", signup_date="2026-09-07", channel="短信", headcount=133, conversion=0.30),
    _row(event_id="E008", signup_date="2026-09-08", channel=None, headcount=96, conversion=0.44),
]


def _version(fields: list[dict[str, Any]], *, version: int = 1, note: str = "初始定义") -> dict[str, Any]:
    return {
        "version": version,
        "created_at": "2026-08-01T09:00:00+08:00",
        "operator": "数据管理员",
        "note": note,
        "source_table": "",
        "fields": fields,
    }


SEED_EVIDENCE: dict[str, list[dict[str, Any]]] = {
    "datastore": [
        {
            "id": 1,
            "status": "已生效",
            "pending": False,
            "abnormal": False,
            "数据集编号": "DATA-0001",
            "数据集名称": "订单交易明细",
            "来源表名": "dws_order_trade",
            "字段列表": "、".join(field["name"] for field in DATASET_1_FIELDS),
            "更新策略": "每日增量",
            "数据采样比例": "10%",
            "生命周期天数": "180",
            "缓存开关": "开启",
            "定义版本": [{**_version(DATASET_1_FIELDS), "source_table": "dws_order_trade"}],
            "样本数据": DATASET_1_SAMPLE,
            "样本说明": "2026 年 9 月首周的脱敏订单小样本，可在训练轮次里逐格修改。",
        },
        {
            "id": 2,
            "status": "已生效",
            "pending": False,
            "abnormal": False,
            "数据集编号": "DATA-0002",
            "数据集名称": "门店销售日汇总",
            "来源表名": "dws_store_sales_day",
            "字段列表": "、".join(field["name"] for field in DATASET_2_FIELDS),
            "更新策略": "每日全量",
            "数据采样比例": "20%",
            "生命周期天数": "365",
            "缓存开关": "开启",
            "定义版本": [{**_version(DATASET_2_FIELDS), "source_table": "dws_store_sales_day"}],
            "样本数据": DATASET_2_SAMPLE,
            "样本说明": "三家门店连续三日的销售小样本。",
        },
        {
            "id": 3,
            "status": "待审核",
            "pending": True,
            "abnormal": False,
            "数据集编号": "DATA-0003",
            "数据集名称": "会员活跃宽表",
            "来源表名": "dws_member_active",
            "字段列表": "、".join(field["name"] for field in DATASET_3_FIELDS),
            "更新策略": "每周全量",
            "数据采样比例": "5%",
            "生命周期天数": "90",
            "缓存开关": "关闭",
            "定义版本": [{**_version(DATASET_3_FIELDS), "source_table": "dws_member_active"}],
            "样本数据": DATASET_3_SAMPLE,
            "样本说明": "会员近 30 天活跃度小样本，未包含活跃评分的派生列。",
        },
        {
            "id": 4,
            "status": "草稿",
            "pending": True,
            "abnormal": False,
            "数据集编号": "DATA-0004",
            "数据集名称": "活动报名明细",
            "来源表名": "ods_event_signup",
            "字段列表": "、".join(field["name"] for field in DATASET_4_FIELDS),
            "更新策略": "手动导入",
            "数据采样比例": "100%",
            "生命周期天数": "30",
            "缓存开关": "关闭",
            "定义版本": [{**_version(DATASET_4_FIELDS), "note": "初始化导入", "source_table": "ods_event_signup"}],
            "样本数据": DATASET_4_SAMPLE,
            "样本说明": "四次投放的活动报名与到场转化小样本。",
        },
    ],
    "forecast": [
        {
            "id": 1,
            "status": "已生效",
            "pending": False,
            "abnormal": False,
            "模型编号": "FORE-0001",
            "模型类型": "线性趋势",
            "训练数据集": "DATA-0001",
            "目标字段": "amount",
            "特征字段": "region,channel",
            "预测步长": "7",
            "置信区间": "95%",
            "评估指标": "MAE",
        },
        {
            "id": 2,
            "status": "待训练",
            "pending": True,
            "abnormal": False,
            "模型编号": "FORE-0002",
            "模型类型": "门店回归",
            "训练数据集": "DATA-0002",
            "目标字段": "sales_amt",
            "特征字段": "weekday,region",
            "预测步长": "14",
            "置信区间": "90%",
            "评估指标": "RMSE",
        },
        {
            "id": 3,
            "status": "草稿",
            "pending": True,
            "abnormal": False,
            "模型编号": "FORE-0003",
            "模型类型": "活跃度分类",
            "训练数据集": "DATA-0003",
            "目标字段": "active_score",
            "特征字段": "tier,login_count",
            "预测步长": "7",
            "置信区间": "95%",
            "评估指标": "AUC",
        },
    ],
}
