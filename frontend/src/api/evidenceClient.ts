type Row = Record<string, any>

/** 把后端 ActionResult 解包；失败时抛出后端给的可读说明。 */
export async function unwrapAction(response: Response): Promise<Row> {
  const payload = (await response.json().catch(() => null)) as Row | null
  if (!response.ok || !payload || payload.ok === false) {
    const message = (payload && (payload.message || payload.detail)) || `接口返回 ${response.status}`
    throw new Error(String(message))
  }
  return payload
}

/** 判断空值（undefined / null / 全空白字符串）。 */
export function isBlank(value: unknown): boolean {
  return value === null || value === undefined || (typeof value === 'string' && value.trim() === '')
}
