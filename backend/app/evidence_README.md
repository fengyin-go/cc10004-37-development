# 趋势预测 · 训练证据解释

趋势预测模块在原有模型维护之外，增加了一条可操作、可核对、可封存的证据解释流程：

```text
发起训练（冻结输入快照）
  → 证据卡串：目标字段 → 特征取值 → 缺值处理 → 评估指标
  → 展开卡片逐行核对样本
  → 点字段跳数据集管理查看「当时定义」
  → 签认：输入版本 + 解释原子封存
```

## 后端

| 文件 | 作用 |
| --- | --- |
| `app/evidence.py` | 解释引擎：只读训练快照，算出四张卡与逐行证据，缺定义时留空并给无法解释依据 |
| `app/evidence_stats.py` | 纯标准库的小样本统计、线性拟合、AUC/F1/准确率与稳定指纹 |
| `app/services/forecast_training.py` | 轮次生命周期：快照冻结、解释迟到裁决、签认幂等、数据集漂移提示 |
| `app/services/datastore.py` | 字段版本链（增/删/改名/类型/缺值策略）与样本修订 |
| `app/seed_evidence.py` | 数据集定义版本、结构化字段、可录入小样本与模型配置 |

### 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/forecast/{id}/rounds` | 轮次列表、当前评估位、已签认轮次 |
| POST | `/api/forecast/{id}/rounds` | 发起训练；`delay_ms` 模拟迟到，`defer` 先冻结快照 |
| GET | `/api/forecast/{id}/rounds/{round_id}` | 快照、四张证据卡、漂移与签认状态 |
| POST | `/api/forecast/{id}/rounds/{round_id}/explain` | 生成解释；已生成或已签认时幂等 |
| POST | `/api/forecast/{id}/rounds/{round_id}/sign` | 签认；重复调用幂等返回同一指纹 |
| PATCH | `/api/forecast/{id}` | 修改目标字段 / 特征 / 评估指标 |
| GET | `/api/datastore/{id}/versions`、`/versions/{version}` | 字段定义历史版本 |
| POST | `/api/datastore/{id}/revise` | 追加字段定义版本 |
| PUT | `/api/datastore/{id}/sample` | 保存可录入小样本（只允许当前定义字段） |

## 规则落点

- **修改小样本产生真实差异**：样本是解释的输入，指标、目标分布、缺值填充值全部按快照样本现算；
  同一份快照重复计算结果一致（签认幂等的基础）。
- **签认原子封存**：签认时把输入版本、模型配置、样本、解释一起深拷贝进 `sealed`，并计算指纹；
  卡片读取优先用封存内容，后续删列、改名、改样本都不倒写。签认前必须先有解释。
- **缺失环节留空**：目标/特征不在快照定义里、缺值策略缺失、填充无法计算、评估口径与目标类型不符时，
  卡片标为「缺失定义」，`basis` 写明无法解释依据，`gaps` 汇总；绝不用数据集当前定义补造历史。
- **并发轮次裁决**：发起顺序决定轮次序号；解释到达时若已有更晚轮次发起，本轮记为「迟到解释」，
  只进历史，不占当前评估位。
- **签认幂等**：同一轮重复签认返回原封存对象与相同指纹，不产生第二份封存。

## 前端

- `frontend/src/views/forecast/index.vue`：模型配置、发起训练（含模拟迟到 / 先冻结快照）、轮次栏、
  证据卡串（可展开样本表、点字段跳数据集管理、缺失依据、签认与漂移提示、快照 JSON）。
- `frontend/src/views/datastore/index.vue`：数据集定义列表、版本切换、字段修订、可录入样本表格；
  支持 `?dataset=CODE&version=N&field=NAME` 定位到证据卡引用的当时定义。

## 验收

```bash
cd backend
.venv/bin/python -m unittest discover -s tests -v
```

前端类型检查：`cd frontend && npm run typecheck`。
