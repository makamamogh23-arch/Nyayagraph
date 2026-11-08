import { useEffect, useMemo, useRef, useState } from 'react'
import type { Done, Source } from '../api'
import { copyText } from '../lib/hooks'
import { citationText, numbering, renderAnswer, type Block } from '../lib/text'

interface Props {
  blocks: Block[]
  sources: Source[]
  streaming: boolean
  selected: number | null
  onCite: (source: number) => void
}

export function AnswerText({ blocks, sources, streaming, selected, onCite }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const html = useMemo(() => renderAnswer(blocks, sources, streaming), [blocks, sources, streaming])

  useEffect(() => {
    ref.current?.querySelectorAll<HTMLButtonElement>('button.cite').forEach((b) => {
      const on = selected != null && Number(b.dataset.src) === selected
      b.classList.toggle('is-selected', on)
      b.setAttribute('aria-pressed', String(on))
    })
  }, [html, selected])

  return (
    <div
      ref={ref}
      className="prose answer-text"
      onClick={(e) => {
        const b = (e.target as HTMLElement).closest<HTMLButtonElement>('button.cite')
        if (b) onCite(Number(b.dataset.src))
      }}
      // markdown-it runs with html:false, so model output cannot inject markup
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

const MODEL_NAMES: Record<string, string> = {
  'claude-opus-5-5': 'Claude Opus 5.5',
  'claude-haiku-5-5': 'Claude Haiku 5.5',
}

export function AnswerFoot({ done, blocks, sources }: { done: Done; blocks: Block[]; sources: Source[] }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    const order = numbering(blocks)
    const body = blocks.map((b) => b.text + (b.cites.length ? b.cites.map((i) => ` [${order.get(i)}]`).join('') : '')).join('')
    const refs = [...order.entries()].sort((a, b) => a[1] - b[1]).map(([i, n]) => `[${n}] ${citationText(sources[i])}`)
    if (await copyText(`${body.trim()}\n\n${refs.join('\n')}\n`)) {
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2000)
    }
  }
  const cost = done.cost_usd < 0.01 ? `$${done.cost_usd.toFixed(4)}` : `$${done.cost_usd.toFixed(2)}`
  return (
    <div className="answer-foot">
      <p className="answer-meta num">
        {MODEL_NAMES[done.model] ?? done.model} · {Math.round(done.seconds)} s · {cost}
        {done.stop_reason === 'max_tokens' && <span className="answer-cut"> · cut off at the length limit</span>}
      </p>
      <button type="button" className="btn btn--quiet btn--sm" onClick={copy}>
        {copied ? 'Copied' : 'Copy answer with citations'}
      </button>
      <span className="visually-hidden" aria-live="polite">{copied ? 'Answer copied' : ''}</span>
    </div>
  )
}

export function AnswerSkeleton() {
  return (
    <div className="skeleton" aria-hidden="true">
      <span style={{ width: '92%' }} />
      <span style={{ width: '97%' }} />
      <span style={{ width: '88%' }} />
      <span style={{ width: '64%' }} />
      <span className="skeleton-gap" />
      <span style={{ width: '95%' }} />
      <span style={{ width: '79%' }} />
    </div>
  )
}
