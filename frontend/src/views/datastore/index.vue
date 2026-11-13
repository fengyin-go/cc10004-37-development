<template>
  <section class="page" data-module="datastore">
    <header class="page-head">
      <div>
        <h2>数据集管理 · 定义与训练样本</h2>
        <p class="page-desc">
          字段定义按版本保存，样本按修订保存。趋势预测的证据卡点字段会落到这里的「当时定义」，
          即使当前版本已删列或改名也能对照。
        </p>
      </div>
      <div class="page-actions">
        <button class="btn" type="button" @click="exportRows">导出数据集管理清单</button>
        <button class="btn primary" type="button" @click="openCreatePanel = !openCreatePanel">登记数据集定义</button>
      </div>
    </header>

    <form v-if="openCreatePanel" class="create-panel" @submit.prevent="createEntry">
      <label v-for="field in createFields" :key="field">
        <span>{{ field }}</span>
        <input v-model="createForm[field]" :placeholder="`请输入${field}`" />
      </label>
      <button class="btn primary" type="submit">登记</button>
      <button class="btn ghost" type="button" @click="openCreatePanel = false">取消</button>
    </form>

    <div v-if="focusNotice" class="focus-banner">
      <span>{{ focusNotice }}</span>
      <button class="link" type="button" @click="clearFocus">退出定位</button>
    </div>

    <p v-if="statusMessage" class="banner">{{ statusMessage }}</p>
    <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>

    <div class="workspace">
      <aside class="dataset-rail">
        <h3>数据集定义</h3>
        <button
          v-for="row in rows"
          :key="String(row.id)"
          type="button"
          class="dataset-item"
          :class="{ active: selectedId === Number(row.id) }"
          @click="selectEntry(row)"
        >
          <span class="dataset-code">{{ row['数据集编号'] }}</span>
          <span class="dataset-name">{{ row['数据集名称'] }}</span>
          <small>{{ row['来源表名'] }} · v{{ row.current_version }} · {{ row.status }}</small>
        </button>
        <p v-if="!rows.length" class="panel-note">暂无数据集定义，可先登记。</p>
      </aside>

      <div class="detail-main">
        <div v-if="!entry" class="empty-state panel-note">选择左侧数据集查看字段定义、版本与样本。</div>
        <template v-else>
          <div class="detail-head">
            <div>
              <strong>{{ entry['数据集编号'] }}</strong>
              <span class="tag">{{ entry['数据集名称'] }}</span>
              <span class="tag soft">来源表 {{ entry['来源表名'] }}</span>
              <span class="tag soft">当前 v{{ entry.current_version }}</span>
            </div>
            <div class="row-actions">
              <button class="link" type="button" @click="runAction('提交审核')">提交审核</button>
              <button class="link" type="button" @click="runAction('生效发布')">生效发布</button>
              <button class="link" type="button" @click="runAction('废弃下线')">废弃下线</button>
            </div>
          </div>

          <div class="version-tabs">
            <button
              v-for="version in versions"
              :key="version.version"
              type="button"
              class="version-tab"
              :class="{ active: viewingVersion === version.version }"
              @click="viewVersion(version.version)"
            >
              v{{ version.version }}
              <small>{{ version.created_at?.slice(0, 10) }}</small>
            </button>
            <span class="version-note">{{ viewingVersionInfo?.note }}</span>
          </div>

          <div class="field-layout">
            <div class="field-table-wrap">
              <h3>
                字段定义
                <small v-if="viewingVersion !== entry.current_version" class="warn-text">
                  （历史版本 v{{ viewingVersion }}，非当前定义）
                </small>
              </h3>
              <table class="sample-table field-table">
                <thead>
                  <tr><th>字段</th><th>类型</th><th>可空</th><th>缺值策略</th><th>说明</th></tr>
                </thead>
                <tbody>
                  <tr
                    v-for="field in viewingFields"
                    :key="field.name"
                    :class="{ highlighted: focus.field === field.name }"
                  >
                    <td><strong>{{ field.name }}</strong></td>
                    <td>{{ field.type }}</td>
                    <td>{{ field.nullable ? '是' : '否' }}</td>
                    <td>{{ field.missing_policy || '—' }}</td>
                    <td>{{ field.description || '—' }}</td>
                  </tr>
                  <tr v-if="!viewingFields.length">
                    <td colspan="5" class="empty-state">这一版还没有字段定义，可新增字段</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div class="revision-panel">
              <h3>修订字段定义</h3>
              <div class="revision-row">
                <select v-model="revisionForm.action">
                  <option v-for="action in revisionActions" :key="action" :value="action">{{ action }}</option>
                </select>
                <input v-model="revisionForm.name" placeholder="字段名" />
                <input v-if="revisionForm.action === '重命名字段'" v-model="revisionForm.new_name" placeholder="新字段名" />
                <select v-if="revisionForm.action === '新增字段' || revisionForm.action === '修改字段类型'" v-model="revisionForm.type">
                  <option v-for="type in fieldTypes" :key="type" :value="type">{{ type }}</option>
                </select>
                <select
                  v-if="revisionForm.action === '新增字段' || revisionForm.action === '修改缺值策略'"
                  v-model="revisionForm.missing_policy"
                >
                  <option v-for="policy in missingPolicies" :key="policy" :value="policy">{{ policy }}</option>
                </select>
                <button class="btn" type="button" @click="applyRevision">提交修订</button>
              </div>
              <p class="panel-hint">修订会追加新版本；已签认的训纟证据卡仍引用旧版本快照。</p>
            </div>
          </div>

          <div class="sample-panel">
            <div class="sample-head">
              <h3>训练样本（{{ sampleRows.length }} 行 · 修订 {{ entry.sample_revision }} 次）</h3>
              <div class="sample-actions">
                <button class="btn" type="button" @click="addSampleRow">加一行</button>
                <button class="btn" type="button" @click="removeSampleRow">删末行</button>
                <button class="btn primary" type="button" @click="saveSample">保存样本</button>
              </div>
            </div>
            <p class="panel-hint">
              直接改格子里的值，保存后下一轮训练会按新样本重新解释；空单元格留空即按缺值策略处理。
              这里改的是样本，不会动字段定义。
            </p>
            <div class="sample-scroll">
              <table class="sample-table">
                <thead>
                  <tr>
                    <th>行</th>
                    <th v-for="field in sampleFields" :key="field.name">{{ field.name }}<small>（{{ field.type }}）</small></th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, rowIndex) in sampleRows" :key="rowIndex">
                    <td>{{ rowIndex + 1 }}</td>
                    <td v-for="field in sampleFields" :key="field.name">
                      <input v-model="row[field.name]" class="cell-input" />
                    </td>
                  </tr>
                  <tr v-if="!sampleRows.length">
                    <td :colspan="sampleFields.length + 1" class="empty-state">还没有样本，可加一行后录入</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </template>
      </div>
    </div>

    <table class="data-table list-table">
      <thead>
        <tr>
          <th v-for="column in columns" :key="column">{{ column }}</th>
          <th>可执行动作</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="row in rows"
          :key="String(row.id)"
          :class="{ active: selectedId === Number(row.id) }"
          @click="selectEntry(row)"
        >
          <td v-for="column in columns" :key="column">{{ row[column] ?? '—' }}</td>
          <td class="row-actions">
            <button class="link" type="button" @click.stop="selectEntry(row)">查看定义</button>
          </td>
        </tr>
        <tr v-if="!rows.length">
          <td :colspan="columns.length + 1" class="empty-state">暂无数据集管理数据</td>
        </tr>
      </tbody>
    </table>

    <footer class="page-foot">
      <span>共 {{ total }} 条数据集定义</span>
      <span v-if="errorMessage" class="error-text">{{ errorMessage }}</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRoute } from 'vue-router'

import { request } from '@/api/client'
import { isBlank, unwrapAction } from '@/api/evidenceClient'
import { readDatastoreQuery } from '@/api/evidence'

type Row = Record<string, any>

const ENDPOINT = '/api/datastore'
const columns = ["数据集编号", "数据集名称", "来源表名", "字段列表", "更新策略", "数据采样比例", "生命周期天数", "缓存开关"]
const createFields = ['数据集编号', '数据集名称', '来源表名']
const revisionActions = ['新增字段', '删除字段', '重命名字段', '修改字段类型', '修改缺值策略']
const fieldTypes = ['文本', '数值', '日期', '布尔']
const missingPolicies = ['丢弃该行', '均值填充', '中位数填充', '众数填充', '零值填充', '不填充']

const route = useRoute()

const rows = ref<Row[]>([])
const total = ref(0)
const entry = ref<Row | null>(null)
const selectedId = ref<number | null>(null)
const versions = ref<Row[]>([])
const viewingVersion = ref<number | null>(null)
const viewingFields = ref<Row[]>([])
const sampleRows = ref<Row[]>([])
const openCreatePanel = ref(false)
const createForm = reactive<Record<string, string>>({})
const revisionForm = reactive<Record<string, string>>({
  action: '新增字段',
  name: '',
  new_name: '',
  type: '文本',
  missing_policy: '不填充',
})
const focus = reactive({ dataset: '', version: null as number | null, field: '' })
const focusNotice = ref('')
const statusMessage = ref('')
const errorMessage = ref('')

const sampleFields = computed<Row[]>(() => {
  if (viewingVersion.value === entry.value?.current_version) return viewingFields.value
  return (versions.value.find((item) => item.version === entry.value?.current_version)?.fields ?? []) as Row[]
})

function currentFields() {
  const current = versions.value.find((item) => item.version === entry.value?.current_version)
  return (current?.fields ?? []) as Row[]
}

const viewingVersionInfo = ref<Row | null>(null)

async function loadList() {
  const response = await request(`${ENDPOINT}?size=200`)
  if (!response.ok) throw new Error('数据集列表读取失败')
  const payload = await response.json()
  rows.value = payload.items ?? []
  total.value = payload.total ?? rows.value.length
}

async function selectEntry(row: Row) {
  selectedId.value = Number(row.id)
  statusMessage.value = ''
  errorMessage.value = ''
  await loadEntry(Number(row.id))
}

async function loadEntry(entryId: number) {
  const response = await request(`${ENDPOINT}/${entryId}`)
  if (!response.ok) throw new Error('数据集定义读取失败')
  entry.value = await response.json()
  await loadVersions(entryId)
  const focused = focus.dataset && focus.dataset === entry.value?.['数据集编号']
  const targetVersion = focused ? focus.version ?? entry.value?.current_version : entry.value?.current_version
  await viewVersion(targetVersion as number, focused ? focus.field : '')
  await loadSample()
}

async function loadVersions(entryId: number) {
  const response = await request(`${ENDPOINT}/${entryId}/versions`)
  if (!response.ok) throw new Error('字段版本读取失败')
  const payload = await response.json()
  versions.value = payload.versions ?? []
}

async function viewVersion(version: number, highlightField = '') {
  if (!entry.value) return
  viewingVersion.value = version
  const item = versions.value.find((candidate) => candidate.version === version)
  if (item) {
    viewingFields.value = item.fields ?? []
    viewingVersionInfo.value = item
    if (highlightField) {
      const present = (item.fields ?? []).some((field: Row) => field.name === highlightField)
      focus.field = highlightField
      focusNotice.value = present
        ? `已定位到 ${entry.value['数据集编号']} v${version} 的字段 ${highlightField}（证据卡引用的当时定义）。`
        : `数据集 ${entry.value['数据集编号']} 的 v${version} 定义里没有字段 ${highlightField}，证据卡按缺失留空。`
    }
    return
  }
  const response = await request(`${ENDPOINT}/${entry.value.id}/versions/${version}`)
  if (!response.ok) throw new Error(`历史版本 v${version} 读取失败`)
  const payload = await response.json()
  viewingFields.value = payload.fields ?? []
  viewingVersionInfo.value = payload
}

async function loadSample() {
  if (!entry.value) return
  const current = currentFields()
  sampleRows.value = (entry.value['样本数据'] ?? []).map((row: Row) => {
    const copy: Row = {}
    for (const field of current) copy[field.name] = row[field.name] ?? null
    return copy
  })
}

function addSampleRow() {
  const row: Row = {}
  for (const field of currentFields()) row[field.name] = null
  sampleRows.value.push(row)
}

function removeSampleRow() {
  sampleRows.value.pop()
}

async function saveSample() {
  if (!entry.value) return
  errorMessage.value = ''
  try {
    const payloadRows = sampleRows.value.map((row) => {
      const clean: Row = {}
      for (const field of currentFields()) {
        const value = row[field.name]
        clean[field.name] = isBlank(value) ? null : value
      }
      return clean
    })
    const response = await request(`${ENDPOINT}/${entry.value.id}/sample`, {
      method: 'PUT',
      body: JSON.stringify({ values: { rows: payloadRows, operator: '值班管理员', note: '手工修订小样本' } }),
    })
    const payload = await unwrapAction(response)
    statusMessage.value = String(payload.message)
    await refreshEntry()
    await loadList()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '样本保存失败'
  }
}

async function applyRevision() {
  if (!entry.value) return
  errorMessage.value = ''
  try {
    const response = await request(`${ENDPOINT}/${entry.value.id}/revise`, {
      method: 'POST',
      body: JSON.stringify({ values: { ...revisionForm, operator: '值班管理员' } }),
    })
    const payload = await unwrapAction(response)
    statusMessage.value = String(payload.message)
    await refreshEntry()
    await loadList()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '字段修订失败'
  }
}

async function runAction(action: string) {
  if (!entry.value) return
  errorMessage.value = ''
  try {
    const response = await request(`${ENDPOINT}/${entry.value.id}/actions`, {
      method: 'POST',
      body: JSON.stringify({ values: { action } }),
    })
    const payload = await unwrapAction(response)
    statusMessage.value = String(payload.message)
    await refreshEntry()
    await loadList()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '状态动作失败'
  }
}

async function refreshEntry() {
  if (!entry.value) return
  await loadEntry(Number(entry.value.id))
}

async function createEntry() {
  errorMessage.value = ''
  try {
    const response = await request(ENDPOINT, {
      method: 'POST',
      body: JSON.stringify({ values: { ...createForm, operator: '值班管理员' } }),
    })
    const payload = await unwrapAction(response)
    statusMessage.value = String(payload.message)
    openCreatePanel.value = false
    createForm['数据集编号'] = ''
    createForm['数据集名称'] = ''
    createForm['来源表名'] = ''
    await loadList()
    if (payload.entry) await selectEntry(payload.entry)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '登记失败'
  }
}

function clearFocus() {
  focus.dataset = ''
  focus.version = null
  focus.field = ''
  focusNotice.value = ''
  if (viewingVersion.value !== null) void viewVersion(viewingVersion.value)
}

function exportRows() {
  window.open(`${ENDPOINT}/export`, '_blank')
}

async function applyRouteFocus() {
  const parsed = readDatastoreQuery(route.fullPath.split('?')[1] ?? '')
  if (!parsed.dataset) return
  focus.dataset = parsed.dataset
  focus.version = parsed.version
  focus.field = parsed.field
  const match = rows.value.find((row) => row['数据集编号'] === parsed.dataset)
  if (!match) {
    errorMessage.value = `数据集 ${parsed.dataset} 不在当前清单里，证据卡仍按快照展示`
    return
  }
  await selectEntry(match)
}

onMounted(async () => {
  try {
    await loadList()
    const parsed = readDatastoreQuery(route.fullPath.split('?')[1] ?? '')
    if (parsed.dataset) {
      await applyRouteFocus()
    } else if (rows.value.length) {
      await selectEntry(rows.value[0])
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '数据集管理数据读取失败'
  }
})

// 从证据卡连续点不同字段时，路由会原地变化，需要重新定位到新的「当时定义」。
watch(
  () => route.fullPath,
  async (next, previous) => {
    const nextQuery = readDatastoreQuery(next.split('?')[1] ?? '')
    const prevQuery = readDatastoreQuery((previous || '').split('?')[1] ?? '')
    if (!nextQuery.dataset) return
    if (nextQuery.dataset === prevQuery.dataset && nextQuery.version === prevQuery.version && nextQuery.field === prevQuery.field) return
    try {
      await applyRouteFocus()
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : '按字段定位失败'
    }
  },
)
</script>

<style scoped>
.create-panel { display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; }
.create-panel label span { display: block; font-size: 12px; color: var(--muted); }
.create-panel input { padding: 5px 8px; border: 1px solid var(--border); border-radius: 6px; }
.focus-banner { background: #eef4ff; border: 1px solid #c7dbff; border-radius: 8px; padding: 8px 12px; font-size: 13px; margin-bottom: 10px; display: flex; justify-content: space-between; }
.banner { background: #fff; border: 1px solid var(--border); border-left: 4px solid var(--brand); border-radius: 6px; padding: 8px 12px; font-size: 13px; margin: 8px 0; }
.workspace { display: flex; gap: 14px; align-items: flex-start; margin: 10px 0 18px; }
.dataset-rail { width: 230px; flex-shrink: 0; background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
.dataset-rail h3 { font-size: 14px; margin: 0 0 8px; }
.dataset-item { display: flex; flex-direction: column; width: 100%; text-align: left; border: 1px solid transparent; background: #f8fafc; border-radius: 6px; padding: 8px; margin-bottom: 6px; cursor: pointer; font-size: 12px; }
.dataset-item.active { border-color: var(--brand); background: #eef4ff; }
.dataset-code { font-weight: 600; font-size: 13px; }
.dataset-name { color: #334155; }
.detail-main { flex: 1; min-width: 0; background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
.detail-head { display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; align-items: center; }
.tag { display: inline-block; font-size: 12px; padding: 2px 8px; border-radius: 10px; background: #e2e8f0; color: #334155; margin-left: 6px; }
.tag.soft { background: #eef2f7; }
.version-tabs { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin: 12px 0 8px; padding-bottom: 8px; border-bottom: 1px solid #eef2f7; }
.version-tab { border: 1px solid var(--border); background: #f8fafc; border-radius: 6px; padding: 3px 10px; cursor: pointer; font-size: 12px; }
.version-tab.active { border-color: var(--brand); background: #eef4ff; color: #1d4ed8; }
.version-tab small { display: block; color: var(--muted); font-size: 10px; }
.version-note { font-size: 12px; color: var(--muted); margin-left: auto; }
.field-layout { display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }
.field-table-wrap { flex: 2; min-width: 320px; }
.revision-panel { flex: 1; min-width: 260px; background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
.revision-panel h3, .field-table-wrap h3, .sample-panel h3 { font-size: 14px; margin: 0 0 8px; }
.revision-row { display: flex; flex-direction: column; gap: 6px; }
.revision-row input, .revision-row select { padding: 5px 8px; border: 1px solid var(--border); border-radius: 6px; }
.panel-hint { font-size: 12px; color: var(--muted); margin: 8px 0 0; }
.sample-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.sample-table th, .sample-table td { border: 1px solid var(--border); padding: 4px 8px; text-align: left; }
.field-table tr.highlighted td { background: #fef9c3; }
.sample-panel { margin-top: 16px; border-top: 1px solid #eef2f7; padding-top: 12px; }
.sample-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
.sample-actions { display: flex; gap: 6px; }
.sample-scroll { overflow-x: auto; margin-top: 8px; }
.cell-input { width: 100%; min-width: 72px; border: 1px solid #e2e8f0; border-radius: 4px; padding: 3px 5px; font-size: 12px; }
.cell-input:focus { border-color: var(--brand); outline: none; }
.list-table { margin-top: 4px; }
tr.active td { background: #eef4ff; }
.panel-note { background: #fff; border: 1px dashed var(--border); border-radius: 8px; padding: 12px; font-size: 13px; color: var(--muted); }
.warn-text { color: #b45309; }
</style>
