<template>
  <section class="page" data-module="forecast">
    <header class="page-head">
      <div>
        <h2>趋势预测 · 训练证据解释</h2>
        <p class="page-desc">
          每轮训练把输入版本与解释封存成证据卡串：目标字段 → 特征取值 → 缺值处理 → 评估指标。
          点字段可回到数据集管理查看当时的定义，展开卡片可逐行核对样本。
        </p>
      </div>
      <div class="page-actions">
        <button class="btn" type="button" @click="reload">刷新轮次</button>
        <button class="btn" type="button" @click="exportRows">导出趋势预测清单</button>
      </div>
    </header>

    <div v-if="!selectedModel" class="empty-state panel-note">下面选一个预测模型，查看它的训练轮次与证据卡。</div>

    <template v-else>
      <div class="model-strip">
        <div class="model-head">
          <strong>{{ selectedModel['模型编号'] }}</strong>
          <span class="tag">{{ selectedModel['模型类型'] }}</span>
          <span class="tag soft">训练数据集 {{ selectedModel['训练数据集'] }}</span>
          <span class="tag soft">评估指标 {{ selectedModel['评估指标'] || '—' }}</span>
          <span class="tag soft">预测步长 {{ selectedModel['预测步长'] || '—' }}</span>
        </div>
        <div class="model-config">
          <label>
            <span>目标字段</span>
            <input v-model="configForm.target" />
          </label>
          <label>
            <span>特征字段（逗号分隔）</span>
            <input v-model="configForm.features" />
          </label>
          <label>
            <span>评估指标</span>
            <select v-model="configForm.metric">
              <option v-for="option in metricOptions" :key="option" :value="option">{{ option }}</option>
            </select>
          </label>
          <button class="btn" type="button" @click="saveConfig">保存配置</button>
        </div>
      </div>

      <div class="train-bar">
        <input v-model="roundNote" class="note-input" placeholder="本轮训练备注（会写进封存记录）" />
        <button class="btn primary" type="button" :disabled="busy" @click="startRound(0)">
          {{ busy ? '处理中…' : '发起训练' }}
        </button>
        <button class="btn" type="button" :disabled="busy" @click="startRound(700)">
          发起训练（模拟迟到 700ms）
        </button>
        <button class="btn" type="button" :disabled="busy" @click="startRound(0, true)">
          发起训练（先冻结快照）
        </button>
        <span class="train-hint">
          当前评估位：第 {{ roundsInfo.current_round_seq || '—' }} 轮（{{ roundsInfo.current_round_status || '无轮次' }}）
        </span>
      </div>

      <p v-if="statusMessage" class="banner" :class="{ warn: statusTone === 'warn' }">{{ statusMessage }}</p>
      <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>

      <div class="rounds-layout">
        <aside class="rounds-rail">
          <h3>训练轮次</h3>
          <button
            v-for="round in roundsInfo.rounds"
            :key="round.round_id"
            type="button"
            class="round-item"
            :class="{ active: selectedRoundId === round.round_id, late: round.late }"
            @click="selectRound(round.round_id)"
          >
            <span class="round-seq">第 {{ round.seq }} 轮</span>
            <span class="round-status" :class="statusClass(round.status)">{{ round.status_text }}</span>
            <small>v{{ round.input_version ?? '—' }} · {{ round.sample_rows }} 行 · {{ round.created_at?.slice(0, 16).replace('T', ' ') }}</small>
            <small v-if="round.signed" class="fingerprint">指纹 {{ round.sealed_fingerprint }}</small>
            <small v-else-if="round.late">迟到，不覆盖新评估</small>
          </button>
          <p v-if="!roundsInfo.rounds.length" class="panel-note">还没有轮次，先发起训练。</p>
        </aside>

        <div class="round-main">
          <div v-if="!selectedRound" class="empty-state panel-note">选择左侧轮次查看证据卡。</div>
          <template v-else>
            <div class="round-head">
              <div>
                <strong>第 {{ selectedRound.summary?.seq }} 轮训练</strong>
                <span class="tag" :class="statusClass(selectedRound.summary?.status || '')">
                  {{ selectedRound.summary?.status_text }}
                </span>
                <span v-if="selectedRound.is_current" class="tag soft">当前评估位</span>
                <span v-if="selectedRound.late" class="tag danger">迟到解释</span>
                <span v-if="selectedRound.sealed" class="tag sealed-tag">已签认封存</span>
              </div>
              <div class="round-actions">
                <button
                  v-if="!selectedRound.sealed"
                  class="btn primary"
                  type="button"
                  :disabled="busy || !selectedRound.explanation"
                  @click="signRound"
                >
                  {{ selectedRound.summary?.signed ? '已签认' : '签认本轮' }}
                </button>
                <button
                  v-if="selectedRound.summary && !selectedRound.explanation"
                  class="btn"
                  type="button"
                  :disabled="busy"
                  @click="explainRound"
                >
                  生成解释
                </button>
              </div>
            </div>

            <div v-if="sealedObjective" class="sealed-box">
              <p>
                <strong>签认封存：</strong>{{ sealedObjective.signed_at }} · {{ sealedObjective.signed_by }} ·
                指纹 <code>{{ selectedRound.sealed?.指纹 }}</code>
              </p>
              <p>
                输入 {{ sealedObjective.input_code }} v{{ sealedObjective.input_version }} ·
                {{ sealedObjective.sample_rows }} 行样本 · 解释指纹
                <code>{{ selectedRound.sealed?.解释指纹 }}</code>
              </p>
              <p v-if="selectedRound.sealed?.迟到" class="warn-text">本轮解释迟到，签认的是历史轮次，不作为当前评估。</p>
            </div>

            <div v-if="driftNotes.length" class="drift-box">
              <strong>数据集漂移提示（不倒写已签认卡片）：</strong>
              <ul>
                <li v-for="(note, index) in driftNotes" :key="index">{{ note.text }}</li>
              </ul>
            </div>

            <p v-if="selectedRound.message" class="banner">{{ selectedRound.message }}</p>

            <div v-if="selectedRound.explanation" class="card-chain">
              <article
                v-for="card in selectedRound.explanation.cards"
                :key="card.key"
                class="evidence-card"
                :class="[`status-${card.status}`, { open: expanded[card.key] }]"
              >
                <button type="button" class="card-head" @click="toggle(card.key)">
                  <span class="step-badge">{{ card.step }}</span>
                  <span class="card-title">{{ card.title }}</span>
                  <span class="card-status">{{ card.status_text }}</span>
                  <span class="expand-hint">{{ expanded[card.key] ? '收起样本' : '展开核对样本' }}</span>
                </button>

                <p class="card-summary">{{ card.summary }}</p>

                <div class="metric-chips">
                  <span
                    v-for="metric in card.metrics"
                    :key="metric.label"
                    class="chip"
                    :class="metric.kind"
                  >
                    {{ metric.label }}：<b>{{ metric.value }}</b>
                  </span>
                </div>

                <div v-if="card.definitions.length" class="def-row">
                  <span class="def-caption">当时定义：</span>
                  <button
                    v-for="def in card.definitions"
                    :key="def.field + (def.present ? '' : '-missing')"
                    type="button"
                    class="def-link"
                    :class="{ missing: !def.present }"
                    @click="openDefinition(def)"
                  >
                    {{ def.field }}
                    <template v-if="def.present">
                      · {{ def.type }} · {{ def.missing_policy || '无策略' }} · v{{ def.version }}
                    </template>
                    <template v-else> · 无当时定义</template>
                  </button>
                </div>

                <ul v-if="card.basis.length" class="basis-list">
                  <li v-for="(item, index) in card.basis" :key="index" :class="{ missing: item.kind === 'missing' }">
                    <span class="basis-tag">{{ item.kind === 'missing' ? '无法解释依据' : '说明' }}</span>
                    {{ item.text }}
                  </li>
                </ul>

                <div v-if="expanded[card.key]" class="sample-box">
                  <div v-if="!card.rows.items.length" class="panel-note">这张卡片没有可展开的样本行。</div>
                  <table v-else class="sample-table">
                    <thead>
                      <tr>
                        <th v-for="column in card.rows.columns" :key="column.key">{{ column.label }}</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="item in card.rows.items" :key="item.row_index" :class="{ excluded: !item.included }">
                        <td v-for="column in card.rows.columns" :key="column.key">
                          {{ item.cells[column.key] }}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                  <p class="sample-foot">
                    灰底行表示不参与本轮训练；样本修改请在
                    <RouterLink :to="`/datastore?dataset=${selectedRound.snapshot?.dataset?.code || ''}`">数据集管理</RouterLink>
                    保存，再发起新一轮训练对比。
                  </p>
                </div>
              </article>
            </div>

            <div v-if="selectedRound.explanation?.gaps.length" class="gaps-box">
              <h3>缺失环节（明确留空，未用当前定义补造）</h3>
              <table class="sample-table">
                <thead>
                  <tr><th>环节</th><th>字段</th><th>无法解释依据</th></tr>
                </thead>
                <tbody>
                  <tr v-for="(gap, index) in selectedRound.explanation.gaps" :key="index">
                    <td>{{ gap.card_title }}</td>
                    <td>{{ gap.field || '—' }}</td>
                    <td>{{ gap.reason }}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <details v-if="selectedRound.snapshot" class="snapshot-box">
              <summary>查看本轮冻结的输入快照（{{ selectedRound.snapshot.dataset?.code }} v{{ selectedRound.snapshot.dataset?.version }}）</summary>
              <pre>{{ snapshotText }}</pre>
            </details>
          </template>
        </div>
      </div>
    </template>

    <header class="page-head sub-head">
      <div>
        <h3>预测模型清单</h3>
        <p class="page-desc">点一行进入训练工作台；列表只展示模型配置。</p>
      </div>
    </header>

    <table class="data-table">
      <thead>
        <tr>
          <th v-for="column in columns" :key="column">{{ column }}</th>
          <th>训练工作台</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="row in rows"
          :key="String(row.id)"
          :class="{ active: selectedModelId === row.id }"
        >
          <td v-for="column in columns" :key="column">{{ row[column] ?? '—' }}</td>
          <td class="row-actions">
            <button class="link" type="button" @click="openModel(row)">查看证据卡</button>
            <button class="link" type="button" @click="quickStart(row)">发起训练</button>
          </td>
        </tr>
        <tr v-if="!rows.length">
          <td :colspan="columns.length + 1" class="empty-state">暂无趋势预测数据</td>
        </tr>
      </tbody>
    </table>

    <footer class="page-foot">
      <span>共 {{ total }} 条预测模型 · 轮次按发起顺序裁决</span>
      <span v-if="errorMessage" class="error-text">{{ errorMessage }}</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'

import { request } from '@/api/client'
import { unwrapAction } from '@/api/evidenceClient'
import type {
  FieldDefinitionRef,
  RoundDetail,
  RoundsPayload,
} from '@/api/evidence'

type Row = Record<string, any>

const ENDPOINT = '/api/forecast'
const columns = ["模型编号", "模型类型", "训练数据集", "目标字段", "特征字段", "预测步长", "置信区间", "评估指标"]
const metricOptions = ['MAE', 'RMSE', 'MAPE', 'AUC', 'F1', '准确率']
const router = useRouter()

const rows = ref<Row[]>([])
const total = ref(0)
const errorMessage = ref('')
const statusMessage = ref('')
const statusTone = ref<'info' | 'warn'>('info')
const busy = ref(false)
const roundNote = ref('')

const selectedModelId = ref<number | null>(null)
const selectedModel = ref<Row | null>(null)
const roundsInfo = ref<RoundsPayload>({
  model: {},
  dataset_code: '',
  rounds: [],
  current_round_id: null,
  current_round_seq: 0,
  current_round_status: '',
  displayed_round_id: null,
  sealed_round_ids: [],
  dataset_version: 0,
})
const selectedRoundId = ref<number | null>(null)
const selectedRound = ref<RoundDetail | null>(null)
const expanded = reactive<Record<string, boolean>>({})
const configForm = reactive({ target: '', features: '', metric: '' })

const sealedObjective = computed(() => {
  const sealed = selectedRound.value?.sealed
  if (!sealed) return null
  const input = sealed['输入版本'] as Record<string, any>
  return {
    signed_at: sealed['签认时间'],
    signed_by: sealed['签认人'],
    input_code: input?.code || '—',
    input_version: input?.version ?? '—',
    sample_rows: sealed['样本行数'],
  }
})

const driftNotes = computed(() => selectedRound.value?.drift || [])
const snapshotText = computed(() => JSON.stringify(selectedRound.value?.snapshot ?? {}, null, 2))

function statusClass(status: string) {
  if (status === '已签认') return 'ok'
  if (status === '迟到解释') return 'danger'
  if (status === '解释生成中') return 'pending'
  if (status === '待签认') return 'ready'
  return ''
}

function toggle(key: string) {
  expanded[key] = !expanded[key]
}

function openDefinition(def: FieldDefinitionRef) {
  const query = new URLSearchParams({
    dataset: def.dataset_code || '',
    version: def.version ? String(def.version) : '',
    field: def.field || '',
  })
  void router.push(`/datastore?${query.toString()}`)
}

async function exportRows() {
  window.open(`${ENDPOINT}/export`, '_blank')
}

async function loadList() {
  const response = await request(`${ENDPOINT}?size=200`)
  if (!response.ok) throw new Error('预测模型列表读取失败')
  const payload = await response.json()
  rows.value = payload.items ?? []
  total.value = payload.total ?? rows.value.length
}

async function openModel(row: Row) {
  selectedModelId.value = Number(row.id)
  selectedModel.value = row
  configForm.target = row['目标字段'] || ''
  configForm.features = row['特征字段'] || ''
  configForm.metric = row['评估指标'] || ''
  expanded['target'] = false
  expanded['feature'] = false
  expanded['missing'] = false
  expanded['metric'] = false
  await loadRounds()
}

async function loadRounds() {
  if (selectedModelId.value === null) return
  const response = await request(`${ENDPOINT}/${selectedModelId.value}/rounds`)
  if (!response.ok) throw new Error('训练轮次读取失败')
  roundsInfo.value = await response.json()
  if (!roundsInfo.value.rounds.length) {
    selectedRoundId.value = null
    selectedRound.value = null
    return
  }
  const preferred = selectedRoundId.value
  const exists = roundsInfo.value.rounds.some((item) => item.round_id === preferred)
  const targetId = exists
    ? preferred
    : roundsInfo.value.current_round_id ?? roundsInfo.value.rounds[0].round_id
  await selectRound(targetId as number)
}

async function selectRound(roundId: number) {
  if (selectedModelId.value === null) return
  selectedRoundId.value = roundId
  const response = await request(`${ENDPOINT}/${selectedModelId.value}/rounds/${roundId}`)
  if (!response.ok) throw new Error('轮次明细读取失败')
  selectedRound.value = await response.json()
  await loadRoundsOnly()
}

async function loadRoundsOnly() {
  if (selectedModelId.value === null) return
  const response = await request(`${ENDPOINT}/${selectedModelId.value}/rounds`)
  if (!response.ok) return
  roundsInfo.value = await response.json()
}

async function startRound(delayMs = 0, defer = false) {
  if (busy.value) return
  busy.value = true
  errorMessage.value = ''
  statusMessage.value = ''
  try {
    if (selectedModelId.value === null) {
      throw new Error('先在下方清单点一个模型')
    }
    const response = await request(`${ENDPOINT}/${selectedModelId.value}/rounds`, {
      method: 'POST',
      body: JSON.stringify({
        values: {
          note: roundNote.value,
          delay_ms: delayMs,
          defer,
          operator: '值班管理员',
        },
      }),
    })
    const payload = await unwrapAction(response)
    statusTone.value = 'info'
    statusMessage.value = String(payload.message || '训练轮次已发起')
    const roundId = payload.entry?.summary?.round_id
    if (roundId) selectedRoundId.value = Number(roundId)
    await refreshModel(selectedModelId.value)
    await loadRounds()
    if (defer && roundId) {
      // 先冻结快照：随后再点「生成解释」也不会重复计算。
      statusTone.value = 'warn'
      statusMessage.value = `${payload.message}；可与其他轮次交错发起，再回来生成解释观察迟到裁决。`
    }
    if (delayMs > 0) {
      window.setTimeout(() => {
        void loadRoundsOnly().then(() => {
          if (selectedRoundId.value) void selectRound(selectedRoundId.value)
        })
      }, delayMs + 400)
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '发起训练失败'
  } finally {
    busy.value = false
  }
}

async function explainRound() {
  if (busy.value || selectedModelId.value === null || !selectedRound.value?.summary) return
  busy.value = true
  errorMessage.value = ''
  try {
    const response = await request(
      `${ENDPOINT}/${selectedModelId.value}/rounds/${selectedRound.value.summary.round_id}/explain`,
      { method: 'POST', body: JSON.stringify({ values: { operator: '解释器' } }) },
    )
    const payload = await unwrapAction(response)
    statusTone.value = payload.entry?.late ? 'warn' : 'info'
    statusMessage.value = payload.message
    await selectRound(selectedRound.value.summary.round_id)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '生成解释失败'
  } finally {
    busy.value = false
  }
}

async function signRound() {
  if (busy.value || selectedModelId.value === null || !selectedRound.value?.summary) return
  busy.value = true
  errorMessage.value = ''
  try {
    const response = await request(
      `${ENDPOINT}/${selectedModelId.value}/rounds/${selectedRound.value.summary.round_id}/sign`,
      { method: 'POST', body: JSON.stringify({ values: { operator: '值班管理员', note: roundNote.value } }) },
    )
    const payload = await unwrapAction(response)
    statusTone.value = 'info'
    statusMessage.value = payload.message
    await selectRound(selectedRound.value.summary.round_id)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '签认失败'
  } finally {
    busy.value = false
  }
}

async function saveConfig() {
  if (selectedModelId.value === null) return
  errorMessage.value = ''
  try {
    const response = await request(`${ENDPOINT}/${selectedModelId.value}`, {
      method: 'PATCH',
      body: JSON.stringify({
        values: {
          '目标字段': configForm.target,
          '特征字段': configForm.features,
          '评估指标': configForm.metric,
        },
      }),
    })
    const payload = await unwrapAction(response)
    statusTone.value = 'info'
    statusMessage.value = payload.message
    await refreshModel(selectedModelId.value)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '保存配置失败'
  }
}

async function refreshModel(modelId: number) {
  const response = await request(`${ENDPOINT}/${modelId}`)
  if (response.ok) {
    const entry = await response.json()
    selectedModel.value = entry
    const index = rows.value.findIndex((item) => Number(item.id) === modelId)
    if (index >= 0) rows.value[index] = entry
  }
}

async function quickStart(row: Row) {
  await openModel(row)
  await startRound(0)
}

async function reload() {
  errorMessage.value = ''
  try {
    await loadList()
    if (selectedModelId.value !== null) {
      await refreshModel(selectedModelId.value)
      await loadRounds()
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '趋势预测数据读取失败'
  }
}

onMounted(async () => {
  try {
    await loadList()
    if (rows.value.length) await openModel(rows.value[0])
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '趋势预测数据读取失败'
  }
})
</script>

<style scoped>
.model-strip { background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 10px; }
.model-head { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
.model-config { display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; }
.model-config label span { display: block; font-size: 12px; color: var(--muted); margin-bottom: 2px; }
.model-config input { padding: 5px 8px; border: 1px solid var(--border); border-radius: 6px; min-width: 130px; }
.train-bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; background: #eef4ff; border: 1px solid #c7dbff; border-radius: 8px; padding: 10px 12px; }
.note-input { padding: 6px 8px; border: 1px solid var(--border); border-radius: 6px; flex: 1; min-width: 220px; }
.train-hint { margin-left: auto; font-size: 12px; color: var(--muted); }
.banner { background: #fff; border: 1px solid var(--border); border-left: 4px solid var(--brand); border-radius: 6px; padding: 8px 12px; font-size: 13px; margin: 10px 0; }
.banner.warn { border-left-color: #d97706; background: #fff8ec; }
.rounds-layout { display: flex; gap: 14px; align-items: flex-start; margin-top: 10px; }
.rounds-rail { width: 240px; flex-shrink: 0; background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
.rounds-rail h3 { font-size: 14px; margin: 0 0 8px; }
.round-item { display: flex; flex-direction: column; gap: 2px; width: 100%; text-align: left; border: 1px solid transparent; background: #f8fafc; border-radius: 6px; padding: 8px; margin-bottom: 6px; cursor: pointer; font-size: 12px; }
.round-item.active { border-color: var(--brand); background: #eef4ff; }
.round-item.late { background: #fff8ec; }
.round-seq { font-weight: 600; font-size: 13px; }
.round-status { align-self: flex-start; border-radius: 10px; padding: 1px 8px; font-size: 11px; background: #e2e8f0; color: #334155; }
.round-status.ok { background: #dcfce7; color: #166534; }
.round-status.ready { background: #dbeafe; color: #1d4ed8; }
.round-status.danger { background: #fee2e2; color: #b91c1c; }
.round-status.pending { background: #fef9c3; color: #854d0e; }
.fingerprint { color: var(--muted); font-family: monospace; }
.round-main { flex: 1; min-width: 0; }
.round-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.round-actions { display: flex; gap: 6px; }
.tag { display: inline-block; font-size: 12px; padding: 2px 8px; border-radius: 10px; background: #e2e8f0; color: #334155; margin-left: 6px; }
.tag.soft { background: #eef2f7; }
.tag.ok { background: #dcfce7; color: #166534; }
.tag.ready { background: #dbeafe; color: #1d4ed8; }
.tag.danger { background: #fee2e2; color: #b91c1c; }
.tag.pending { background: #fef9c3; color: #854d0e; }
.tag.sealed-tag { background: #ede9fe; color: #5b21b6; }
.sealed-box { background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 8px; padding: 8px 12px; font-size: 13px; margin-top: 8px; }
.sealed-box code { font-family: monospace; background: #fff; padding: 0 4px; border-radius: 4px; }
.sealed-box p { margin: 2px 0; }
.warn-text { color: #b45309; }
.drift-box { background: #fff8ec; border: 1px solid #fde68a; border-radius: 8px; padding: 8px 12px; font-size: 13px; margin-top: 8px; }
.drift-box ul { margin: 4px 0 0; padding-left: 18px; }
.card-chain { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
.evidence-card { background: #fff; border: 1px solid var(--border); border-left: 4px solid var(--brand); border-radius: 8px; }
.evidence-card.status-partial { border-left-color: #d97706; }
.evidence-card.status-missing { border-left-color: #b42318; }
.card-head { display: flex; align-items: center; gap: 8px; width: 100%; background: none; border: none; border-bottom: 1px solid #eef2f7; padding: 8px 12px; cursor: pointer; text-align: left; }
.step-badge { width: 22px; height: 22px; border-radius: 50%; background: var(--brand); color: #fff; display: flex; align-items: center; justify-content: center; font-size: 12px; flex-shrink: 0; }
.card-title { font-weight: 600; font-size: 14px; }
.card-status { font-size: 12px; color: var(--muted); border: 1px solid var(--border); border-radius: 10px; padding: 1px 8px; }
.expand-hint { margin-left: auto; font-size: 12px; color: var(--brand); }
.card-summary { margin: 8px 12px; font-size: 13px; line-height: 1.6; }
.metric-chips { display: flex; gap: 6px; flex-wrap: wrap; padding: 0 12px 8px; }
.chip { font-size: 12px; background: #f1f5f9; border-radius: 10px; padding: 2px 9px; color: #334155; }
.chip.metric { background: #eff6ff; color: #1d4ed8; }
.chip.baseline { background: #f5f3ff; color: #5b21b6; }
.chip b { font-weight: 600; }
.def-row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; padding: 0 12px 8px; }
.def-caption { font-size: 12px; color: var(--muted); }
.def-link { font-size: 12px; border: 1px solid #bfdbfe; background: #eff6ff; color: #1d4ed8; border-radius: 12px; padding: 2px 10px; cursor: pointer; }
.def-link:hover { background: #dbeafe; }
.def-link.missing { border-color: #fecaca; background: #fef2f2; color: #b91c1c; }
.basis-list { list-style: none; margin: 0; padding: 0 12px 10px; display: flex; flex-direction: column; gap: 4px; }
.basis-list li { font-size: 12px; color: #475569; }
.basis-list li.missing { color: #b42318; background: #fef2f2; border-radius: 6px; padding: 4px 8px; }
.basis-tag { font-weight: 600; margin-right: 4px; }
.sample-box { border-top: 1px dashed var(--border); padding: 8px 12px 12px; overflow-x: auto; }
.sample-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.sample-table th, .sample-table td { border: 1px solid var(--border); padding: 4px 8px; text-align: left; }
.sample-table tr.excluded td { background: #f1f5f9; color: #94a3b8; }
.sample-foot { font-size: 12px; color: var(--muted); margin: 6px 0 0; }
.gaps-box { background: #fff; border: 1px solid #fecaca; border-radius: 8px; padding: 10px 12px; margin-top: 10px; }
.gaps-box h3 { font-size: 14px; margin: 0 0 8px; color: #b42318; }
.snapshot-box { margin-top: 10px; background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; font-size: 12px; }
.snapshot-box pre { max-height: 260px; overflow: auto; background: #f8fafc; padding: 8px; border-radius: 6px; }
.sub-head { margin-top: 24px; }
tr.active td { background: #eef4ff; }
.panel-note { background: #fff; border: 1px dashed var(--border); border-radius: 8px; padding: 12px; font-size: 13px; color: var(--muted); }
</style>
