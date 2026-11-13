/** 训练证据解释的类型定义与跨模块跳转辅助。 */

export type EvidenceCardStatus = 'explained' | 'partial' | 'missing'

export interface FieldDefinitionRef {
  dataset_id: number | null
  dataset_code: string
  dataset_name: string
  version: number | null
  version_created_at: string | null
  field: string
  link: string
  present: boolean
  type: string
  nullable: boolean | null
  description: string
  missing_policy: string
}

export interface EvidenceMetric {
  label: string
  value: string
  kind: 'stat' | 'metric' | 'baseline'
  hint?: string
}

export interface EvidenceBasisItem {
  kind: 'missing' | 'note'
  field?: string
  text: string
}

export interface EvidenceRowColumn {
  key: string
  label: string
  type: string
}

export interface EvidenceRowItem {
  row_index: number
  cells: Record<string, string | number>
  included: boolean
}

export interface EvidenceCard {
  key: 'target' | 'feature' | 'missing' | 'metric'
  step: number
  title: string
  status: EvidenceCardStatus
  status_text: string
  summary: string
  definitions: FieldDefinitionRef[]
  metrics: EvidenceMetric[]
  basis: EvidenceBasisItem[]
  rows: { columns: EvidenceRowColumn[]; items: EvidenceRowItem[] }
  extras: Record<string, unknown>
}

export interface EvidenceExplanation {
  cards: EvidenceCard[]
  gaps: { card: string; card_title: string; field: string; reason: string }[]
  metric_values: Record<string, string | number>
  fit: Record<string, unknown>
}

export interface RoundSummary {
  round_id: number
  seq: number
  status: string
  status_text: string
  created_at: string
  operator: string
  note: string
  late: boolean
  effective: boolean
  signed: boolean
  sealed_fingerprint: string | null
  signed_at: string | null
  signed_by: string | null
  metric_values: Record<string, string | number>
  gaps: number
  input_version: number | null
  input_dataset: string
  sample_rows: number
}

export interface SealedPayload {
  轮次: number
  轮次编号: number
  签认时间: string
  签认人: string
  签认说明: string
  输入版本: Record<string, unknown>
  模型配置: Record<string, unknown>
  样本行数: number
  样本: Record<string, unknown>[]
  样本指纹: string
  输入指纹: string
  解释: EvidenceExplanation
  解释指纹: string
  指纹: string
  迟到: boolean
}

export interface RoundDetail {
  summary: RoundSummary | null
  snapshot: Record<string, any> | null
  explanation: EvidenceExplanation | null
  sealed: SealedPayload | null
  is_current: boolean
  late: boolean
  drift: { kind: string; text: string }[]
  superseded_by: number | null
  message?: string
  model?: Record<string, string | number | boolean | null>
}

export interface RoundsPayload {
  model: Record<string, any>
  dataset_code: string
  rounds: RoundSummary[]
  current_round_id: number | null
  current_round_seq: number
  current_round_status: string
  displayed_round_id: number | null
  sealed_round_ids: number[]
  dataset_version: number
}

/** 证据卡字段 -> 数据集管理「当时定义」的跳转地址。 */
export function definitionLink(def: FieldDefinitionRef | undefined): string {
  if (!def) return '/datastore'
  return def.link || `/datastore?dataset=${encodeURIComponent(def.dataset_code || '')}`
}

/** 从 URL 读取数据集页需要的定位参数。 */
export function readDatastoreQuery(search: string): { dataset: string; version: number | null; field: string } {
  const params = new URLSearchParams(search)
  const versionText = params.get('version')
  return {
    dataset: params.get('dataset') || '',
    version: versionText ? Number(versionText) : null,
    field: params.get('field') || '',
  }
}
