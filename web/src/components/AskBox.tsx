import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import type { Mode } from '../api'
import { useMediaQuery, useSlashFocus } from '../lib/hooks'

const MAX = 2000

/** Stacked pages: reads every source closely. */
function ThoroughIcon() {
  return (
    <svg className="mode-icon" viewBox="0 0 16 16" aria-hidden="true">
      <rect x="2.75" y="4.25" width="8.5" height="9.5" rx="1.5" />
      <path d="M5 2.25h7.25c.83 0 1.5.67 1.5 1.5v8.25" />
      <path d="M5 8h4M5 10.5h2.5" />
    </svg>
  )
}

/** Bolt: fast and cheap. */
function QuickIcon() {
  return (
    <svg className="mode-icon" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8.9 1.75 3.5 9.1h4.1l-.75 5.15L12.5 6.9H8.35l.55-5.15Z" />
    </svg>
  )
}

// Times measured on this machine: Opus 18-36 s, Haiku 8-26 s per answer; costs from the benchmark runs.
const MODES: { value: Mode; name: string; caption: string; Icon: () => React.JSX.Element }[] = [
  { value: 'full', name: 'Thorough', caption: 'Claude Opus 5.5 · about $0.08 · 20–30 s', Icon: ThoroughIcon },
  { value: 'quick', name: 'Quick', caption: 'Claude Haiku 5.5 · about $0.002 · around 10 s', Icon: QuickIcon },
]

interface Props {
  initial?: string
  mode: Mode
  running: boolean
  compact?: boolean
  autoFocus?: boolean
  placeholder?: string
  onAsk: (question: string, mode: Mode) => void
  onStop: () => void
}

export default function AskBox({ initial = '', mode: initialMode, running, compact, autoFocus, placeholder, onAsk, onStop }: Props) {
  const [text, setText] = useState(initial)
  const [mode, setMode] = useState<Mode>(initialMode)
  const ref = useRef<HTMLTextAreaElement>(null)
  const hintId = useId()
  const modeName = useId()
  const narrow = useMediaQuery('(max-width: 520px)')
  useSlashFocus(ref)

  // grow with the question, up to ~6 lines
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 168)}px`
  }, [text])

  useEffect(() => {
    // only with a precise pointer: on phones, focusing on load would throw up the keyboard
    if (autoFocus && window.matchMedia('(pointer: fine)').matches) ref.current?.focus()
  }, [autoFocus])

  const q = text.trim()
  const canAsk = q.length >= 3 && q.length <= MAX && !running

  const submit = () => {
    if (canAsk) onAsk(q, mode)
  }

  return (
    <form
      className={`askbox${compact ? ' askbox--compact' : ''}`}
      role="search"
      aria-label="Ask a question"
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
    >
      <label className="visually-hidden" htmlFor="question">Your question</label>
      <div className="askbox-field">
        <textarea
          id="question"
          ref={ref}
          rows={1}
          value={text}
          maxLength={MAX}
          placeholder={placeholder ?? (narrow ? "Ask a criminal-law question" : "e.g. When does a dying declaration need corroboration?")}
          aria-describedby={hintId}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              submit()
            }
          }}
        />
        {running ? (
          <button type="button" className="btn btn--quiet askbox-go" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" className="btn btn--primary askbox-go" disabled={!canAsk}>
            Ask
          </button>
        )}
      </div>
      <div className="askbox-foot">
        <div className="modes-row">
          <fieldset className="modes" data-mode={mode}>
            <legend className="visually-hidden">Answer mode</legend>
            <span className="modes-thumb" aria-hidden="true" />
            {MODES.map((m) => (
              <label key={m.value} className="mode">
                <input
                  type="radio"
                  className="visually-hidden"
                  name={modeName}
                  value={m.value}
                  checked={mode === m.value}
                  aria-describedby={`${modeName}-${m.value}`}
                  onChange={() => setMode(m.value)}
                />
                <m.Icon />
                <span className="mode-name">{m.name}</span>
              </label>
            ))}
          </fieldset>
          <p className="mode-caption">
            {MODES.map((m) => (
              <span key={m.value} id={`${modeName}-${m.value}`} className="mode-caption-item" data-on={mode === m.value || undefined}>
                {m.caption}
              </span>
            ))}
          </p>
        </div>
        <p className="askbox-hint" id={hintId}>
          {q.length > MAX - 200 ? `${MAX - q.length} characters left` : <>Enter to ask · Shift+Enter for a new line · <kbd>/</kbd> to focus</>}
        </p>
      </div>
    </form>
  )
}
