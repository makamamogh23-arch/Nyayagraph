import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import type { Source } from '../api'
import { copyText } from '../lib/hooks'
import { caseHref, citationText, numbering, quoteRanges, quotesBySource, type Block } from '../lib/text'

interface Props {
  sources: Source[]
  blocks: Block[]
  selected: number | null
  onSelect: (source: number | null) => void
  answering: boolean
}

export default function Sources({ sources, blocks, selected, onSelect, answering }: Props) {
  const order = useMemo(() => numbering(blocks), [blocks])
  const quotes = useMemo(() => quotesBySource(blocks, sources), [blocks, sources])
  const cited = [...order.entries()].sort((a, b) => a[1] - b[1]).map(([i]) => i)
  const rest = sources.map((s) => s.index).filter((i) => !order.has(i))
  const [restOpen, setRestOpen] = useState(false)
  // a citation picked from the answer may point at a source not yet shown
  const showRest = restOpen || (selected != null && !order.has(selected))
  const setShowRest = setRestOpen

  return (
    <div className="sources">
      <h2 className="sources-title">
        Sources{' '}
        <span className="sources-count num">
          {cited.length > 0 ? `${cited.length} cited` : answering ? 'waiting for citations' : 'none cited'}
          {' · '}
          {sources.length} read
        </span>
      </h2>
      {cited.length > 0 && (
        <ol className="source-list">
          {cited.map((i) => (
            <SourceItem key={i} source={sources[i]} n={order.get(i)!} quotes={quotes.get(i) ?? []}
              selected={selected === i} onSelect={onSelect} />
          ))}
        </ol>
      )}
      {rest.length > 0 && (
        <div className="sources-rest">
          <button type="button" className="sources-rest-toggle" aria-expanded={showRest} onClick={() => setShowRest((v) => !v)}>
            {showRest ? 'Hide' : 'Show'} {rest.length} {cited.length ? 'more paragraphs read but not cited' : 'paragraphs read'}
          </button>
          {showRest && (
            <ul className="source-list source-list--rest">
              {rest.map((i) => (
                <SourceItem key={i} source={sources[i]} n={null} quotes={[]} selected={selected === i} onSelect={onSelect} />
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

interface ItemProps {
  source: Source
  n: number | null
  quotes: string[]
  selected: boolean
  onSelect: (source: number | null) => void
}

function SourceItem({ source: s, n, quotes, selected, onSelect }: ItemProps) {
  const [open, setOpen] = useState(selected)
  const [prevSelected, setPrevSelected] = useState(selected)
  if (selected !== prevSelected) {
    // picked from the answer: open it (state adjusted during render, not in an effect)
    setPrevSelected(selected)
    if (selected) setOpen(true)
  }
  const [copied, setCopied] = useState(false)
  const head = useRef<HTMLButtonElement>(null)
  const item = useRef<HTMLLIElement>(null)
  const bodyId = `src-${s.index}`

  useEffect(() => {
    if (!selected) return
    item.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    head.current?.focus({ preventScroll: true })
  }, [selected])

  const ranges = useMemo(() => quoteRanges(s.text, quotes), [s.text, quotes])
  const copy = async () => {
    if (await copyText(citationText(s))) {
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <li ref={item} className={`source${selected ? ' is-selected' : ''}${open ? ' is-open' : ''}`}>
      <button
        ref={head}
        type="button"
        className="source-head"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => {
          setOpen((v) => !v)
          onSelect(open ? null : s.index)
        }}
      >
        {n != null ? <span className="source-n num" aria-label={`Source ${n}`}>{n}</span> : <span className="source-n source-n--none" aria-hidden="true" />}
        <span className="source-main">
          <span className="source-case">{s.title}</span>
          <span className="source-meta num">
            {s.case_id} · ¶ {s.para_no}
            {s.opinion > 0 && ` · separate opinion ${s.opinion}`}
          </span>
        </span>
        <span className="source-chevron" aria-hidden="true" />
      </button>
      {open && (
        <div className="source-body" id={bodyId}>
          {s.precedent && s.cited_from && (
            <p className="source-why">A precedent, cited at {s.cited_from}.</p>
          )}
          <p className="source-text">
            <Highlighted text={s.text} ranges={ranges} />
          </p>
          {ranges.length > 0 && <p className="source-legend"><mark>Highlighted</mark> words are quoted in the answer.</p>}
          <div className="source-actions">
            <Link className="btn btn--quiet btn--sm" to={caseHref(s.case_id, s.opinion, s.para_no)}>
              Read in the judgment
            </Link>
            <button type="button" className="btn btn--quiet btn--sm" onClick={copy}>
              {copied ? 'Copied' : 'Copy citation'}
            </button>
            <span className="visually-hidden" aria-live="polite">{copied ? 'Citation copied' : ''}</span>
          </div>
        </div>
      )}
    </li>
  )
}

function Highlighted({ text, ranges }: { text: string; ranges: [number, number][] }) {
  if (!ranges.length) return <>{text}</>
  const parts: React.ReactNode[] = []
  let at = 0
  ranges.forEach(([a, b], k) => {
    if (a > at) parts.push(text.slice(at, a))
    parts.push(<mark key={k}>{text.slice(a, b)}</mark>)
    at = b
  })
  if (at < text.length) parts.push(text.slice(at))
  return <>{parts}</>
}
