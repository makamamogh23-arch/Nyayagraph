// Typed client for api/server.py.

export type Mode = 'full' | 'quick'

export interface Meta {
  cases: number
  paragraphs: number
  precedents: number
  years: [number, number]
  court: string
  examples: string[]
  modes: Record<Mode, { model: string; price_per_mtok: [number, number] | null }>
}

export interface Source {
  index: number
  pid: string
  label: string
  case_id: string
  opinion: number
  para_no: number
  title: string
  year: number | null
  report: string | null
  text: string
  why: string[]
  precedent: boolean
  cited_from: string | null
}

export interface NamedCase {
  case_id: string
  label: string
  title: string | null
}

export interface Context {
  provisions: string[]
  provision_map: string[]
  related_map?: string[]
  named_cases: NamedCase[]
  sources: Source[]
  seconds: number
}

export interface Done {
  model: string
  usage: { in: number; out: number } | null
  cost_usd: number
  seconds: number
  stop_reason: string
}

export interface ApiError {
  code: string
  message: string
}

export type Stage = 'analysing' | 'retrieving' | 'answering'

export type AskEvent =
  | { type: 'stage'; data: { step: Stage } }
  | { type: 'context'; data: Context }
  | { type: 'text'; data: { block: number; text: string } }
  | { type: 'cite'; data: { block: number; source: number; cited_text: string } }
  | { type: 'done'; data: Done }
  | { type: 'error'; data: ApiError }

export async function getMeta(signal?: AbortSignal): Promise<Meta> {
  const r = await fetch('/api/meta', { signal })
  if (!r.ok) throw new Error(`meta ${r.status}`)
  return r.json()
}

/** POST /api/ask and yield its server-sent events as they arrive. */
export async function* ask(question: string, mode: Mode, signal: AbortSignal): AsyncGenerator<AskEvent> {
  let r: Response
  try {
    r = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question, mode }),
      signal,
    })
  } catch (e) {
    if ((e as Error).name === 'AbortError') return
    yield { type: 'error', data: { code: 'offline', message: 'Could not reach the Nyayagraph server. Is it running?' } }
    return
  }
  if (!r.ok || !r.body) {
    const message = r.status === 422 ? 'Questions need between 3 and 2,000 characters.' : `The server answered ${r.status}.`
    yield { type: 'error', data: { code: `http_${r.status}`, message } }
    return
  }
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader()
  let buf = ''
  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buf += value
      let cut: number
      while ((cut = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, cut)
        buf = buf.slice(cut + 2)
        let type = 'message'
        let data = ''
        for (const line of chunk.split('\n')) {
          if (line.startsWith('event: ')) type = line.slice(7)
          else if (line.startsWith('data: ')) data += line.slice(6)
        }
        if (data) yield { type, data: JSON.parse(data) } as AskEvent
      }
    }
  } catch (e) {
    if ((e as Error).name === 'AbortError') return
    yield { type: 'error', data: { code: 'stream_broken', message: 'The connection dropped before the answer finished.' } }
  }
}

export interface Provision {
  act: string
  section: string
  paras: string[]
  n: number
  maps_to: { act: string; section: string; sub: string }[]
}

export interface CaseLink {
  case_id: string
  label: string
  title: string | null
  year: number | null
  in_corpus?: boolean
  paras: string[]
}

export interface Paragraph {
  pid: string
  opinion: number
  para_no: number
  text: string
}

/** Entities from the fine-tuned NER pass (ner/corpus_entities.py); empty lists until it is loaded. */
export interface AppealFrom {
  court: string
  bench: string | null
  date: string | null
  case_numbers: string[]
  source: string
}

export type Side = 'appellant' | 'respondent' | 'parties' | 'intervenor' | 'complainant' | 'amicus' | 'other'

export interface Counsel {
  name: string
  side: Side
  senior: boolean
  cases: number
}

export interface Witness {
  name: string
  pw: number
  paras: string[]
}

export interface CaseDetail {
  case_id: string
  in_corpus: boolean
  label: string
  title: string | null
  date: string | null
  year: number | null
  report: string | null
  disposal: string | null
  parallel: string[]
  judges: { authored: string[]; bench: string[] }
  provisions: Provision[]
  cites: CaseLink[]
  cited_by: CaseLink[]
  appeal_from?: AppealFrom[]
  counsel?: Counsel[]
  witnesses?: Witness[]
  paragraphs: Paragraph[]
  opinions: number[]
}

export class HttpError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function getCase(id: string, signal?: AbortSignal): Promise<CaseDetail> {
  const r = await fetch(`/api/case/${encodeURIComponent(id)}`, { signal })
  if (!r.ok) {
    let detail = `The server answered ${r.status}.`
    try {
      detail = (await r.json()).detail ?? detail
    } catch {
      /* not JSON */
    }
    throw new HttpError(r.status, detail)
  }
  return r.json()
}
