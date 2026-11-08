import MarkdownIt from 'markdown-it'
import type { Source } from '../api'

/** Answer text arrives as content blocks; a block that cites sources carries their indices. */
export interface Block {
  text: string
  cites: number[]
}

// html:false escapes any raw HTML in model output; linkify off (answers do not need links).
const md = new MarkdownIt({ html: false, linkify: false, typographer: false, breaks: false })

const squash = (t: string) => t.replace(/\W+/g, ' ').trim().toLowerCase()

/** Source index -> display number, in order of first citation. */
export function numbering(blocks: Block[]): Map<number, number> {
  const order = new Map<number, number>()
  for (const b of blocks) for (const s of b.cites) if (!order.has(s)) order.set(s, order.size + 1)
  return order
}

/** True when the block's own words appear verbatim in one of the sources it cites: a quotation. */
export function isVerbatim(text: string, cites: number[], sources: Source[]): boolean {
  const core = text.trim()
  if (core.length <= 20 || !cites.length) return false
  const q = squash(core)
  return cites.some((i) => sources[i] && squash(sources[i].text).includes(q))
}

const OPEN = ''
const CLOSE = ''

/** Render the answer to HTML, with citation markers as buttons (data-src = source index). */
export function renderAnswer(blocks: Block[], sources: Source[], streaming: boolean): string {
  const order = numbering(blocks)
  // A cited block followed by one opening with punctuation: move the punctuation before the marker, so a line
  // never starts with ", and" after a citation.
  const texts = blocks.map((b) => b.text)
  const tails = blocks.map(() => '')
  for (let k = 0; k < blocks.length - 1; k++) {
    if (!blocks[k].cites.length) continue
    const m = texts[k + 1].match(/^[,.;:)\]!?’”]+/)
    if (m) {
      tails[k] = m[0]
      texts[k + 1] = texts[k + 1].slice(m[0].length)
    }
  }
  let out = ''
  blocks.forEach((b, k) => {
    let text = texts[k]
    const core = text.trim()
    const startsStructure = /^(?:[-*+]\s|\d+[.)]\s|#{1,6}\s|>)/.test(core)
    if (!startsStructure && !/^["“]/.test(core) && isVerbatim(text, b.cites, sources)) {
      const lead = text.slice(0, text.length - text.trimStart().length)
      const trail = text.slice(text.trimEnd().length)
      text = `${lead}“${core}”${trail}`
    }
    // punctuation goes after the closing quote and before the marker: “...”, [1]
    if (tails[k]) text = text.replace(/\s*$/, (ws) => tails[k] + ws)
    out += text
    if (b.cites.length) {
      const ns = [...new Set(b.cites)].sort((a, c) => order.get(a)! - order.get(c)!)
      out += OPEN + ns.join(',') + CLOSE
    }
  })
  let html = md.render(out)
  html = html.replace(new RegExp(`${OPEN}([\\d,]+)${CLOSE}`, 'g'), (_, list: string) =>
    list
      .split(',')
      .map((s) => {
        const i = Number(s)
        const n = order.get(i)
        const src = sources[i]
        const name = src ? `${src.title}, paragraph ${src.para_no}` : ''
        return `<button type="button" class="cite" data-src="${i}" aria-label="Source ${n}: ${escapeAttr(name)}">${n}</button>`
      })
      .join(''),
  )
  if (streaming) html = html.replace(/<\/(p|li)>\s*$/, '<span class="caret" aria-hidden="true"></span></$1>')
  return html
}

function escapeAttr(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;')
}

/** Character ranges in `text` matching the quoted phrases (whitespace and punctuation tolerant). */
export function quoteRanges(text: string, phrases: string[]): [number, number][] {
  const ranges: [number, number][] = []
  for (const p of phrases) {
    const words = p.match(/[\p{L}\p{N}]+/gu)
    if (!words || words.length < 3) continue
    const re = new RegExp(words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('[^\\p{L}\\p{N}]+'), 'giu')
    const m = re.exec(text)
    if (m) ranges.push([m.index, m.index + m[0].length])
  }
  ranges.sort((a, b) => a[0] - b[0])
  const merged: [number, number][] = []
  for (const r of ranges) {
    const last = merged[merged.length - 1]
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1])
    else merged.push([...r])
  }
  return merged
}

/** For each source index, the answer passages that quote it word for word. */
export function quotesBySource(blocks: Block[], sources: Source[]): Map<number, string[]> {
  const out = new Map<number, string[]>()
  for (const b of blocks) {
    for (const i of new Set(b.cites)) {
      if (isVerbatim(b.text, [i], sources)) {
        const list = out.get(i) ?? []
        list.push(b.text.trim().replace(/^["“]|["”]$/g, ''))
        out.set(i, list)
      }
    }
  }
  return out
}

const ACTS: Record<string, string> = {
  'Indian Penal Code, 1860': 'IPC',
  'Code of Criminal Procedure, 1973': 'CrPC',
  'Indian Evidence Act, 1872': 'Evidence Act',
  'Bharatiya Nyaya Sanhita, 2023': 'BNS',
  'Bharatiya Nagarik Suraksha Sanhita, 2023': 'BNSS',
  'Bharatiya Sakshya Adhiniyam, 2023': 'BSA',
  'Constitution of India': 'Constitution',
}

export const shortAct = (act: string) => ACTS[act] ?? act

/** 'Code of Criminal Procedure, 1973 s.482 = Bharatiya Nagarik Suraksha Sanhita, 2023 s.528' -> parts. */
export function parseCorrespondence(line: string): { old: string; next: string; label?: string } | null {
  const m = line.match(/^(.*?) s\.(\S+)(?: \(([^)]+)\))? = (.*?) s\.(\S+(?: proviso)?)$/)
  if (!m) return null
  return { old: `${shortAct(m[1])} s.${m[2]}`, next: `${shortAct(m[4])} s.${m[5]}`, label: m[3] }
}

export function provisionLabel(p: string): string {
  const m = p.match(/^(.*) s\.(.+)$/)
  if (!m) return p
  return m[1] === 'Constitution of India' ? `Article ${m[2]}` : `${shortAct(m[1])} s.${m[2]}`
}

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

/** '2016-06-29' -> '29 June 2016' */
export function formatDate(iso: string | null): string | null {
  if (!iso) return null
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})/)
  if (!m) return iso
  return `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]} ${m[1]}`
}

/** Ready-to-paste reference: 'Satye Singh v. State of Uttarakhand, 2022 INSC 165, ¶ 12'. */
export function citationText(s: Pick<Source, 'title' | 'case_id' | 'para_no' | 'opinion' | 'report'>): string {
  const op = s.opinion > 0 ? ` (separate opinion ${s.opinion})` : ''
  const report = s.report ? ` : ${s.report}` : ''
  return `${s.title}, ${s.case_id}${report}, ¶ ${s.para_no}${op}`
}

export const nf = new Intl.NumberFormat('en-IN')

export function caseHref(caseId: string, opinion?: number, para?: number): string {
  const base = `/case/${encodeURIComponent(caseId)}`
  return para != null ? `${base}#p-${opinion ?? 0}-${para}` : base
}
