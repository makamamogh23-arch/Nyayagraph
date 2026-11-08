import { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef, type ReactNode } from 'react'
import { ask, type ApiError, type Context, type Done, type Mode, type Stage } from './api'
import type { Block } from './lib/text'

export type Status = 'idle' | 'running' | 'done' | 'error' | 'stopped'

export interface Session {
  question: string
  mode: Mode
  status: Status
  stage: Stage | null
  context: Context | null
  blocks: Block[]
  done: Done | null
  error: ApiError | null
  selected: number | null // source index highlighted from a citation
}

const empty: Session = {
  question: '', mode: 'full', status: 'idle', stage: null, context: null, blocks: [], done: null, error: null,
  selected: null,
}

type Action =
  | { type: 'start'; question: string; mode: Mode }
  | { type: 'stage'; stage: Stage }
  | { type: 'context'; context: Context }
  | { type: 'text'; block: number; text: string }
  | { type: 'cite'; block: number; source: number }
  | { type: 'done'; done: Done }
  | { type: 'error'; error: ApiError }
  | { type: 'stopped' }
  | { type: 'select'; source: number | null }
  | { type: 'reset' }

function reducer(s: Session, a: Action): Session {
  switch (a.type) {
    case 'start':
      return { ...empty, question: a.question, mode: a.mode, status: 'running', stage: 'analysing' }
    case 'stage':
      return { ...s, stage: a.stage }
    case 'context':
      return { ...s, context: a.context }
    case 'text': {
      const blocks = s.blocks.slice()
      while (blocks.length <= a.block) blocks.push({ text: '', cites: [] })
      blocks[a.block] = { ...blocks[a.block], text: blocks[a.block].text + a.text }
      return { ...s, blocks }
    }
    case 'cite': {
      const blocks = s.blocks.slice()
      while (blocks.length <= a.block) blocks.push({ text: '', cites: [] })
      const b = blocks[a.block]
      if (!b.cites.includes(a.source)) blocks[a.block] = { ...b, cites: [...b.cites, a.source] }
      return { ...s, blocks }
    }
    case 'done':
      return { ...s, status: 'done', stage: null, done: a.done }
    case 'error':
      return { ...s, status: 'error', stage: null, error: a.error }
    case 'stopped':
      return { ...s, status: 'stopped', stage: null }
    case 'select':
      return { ...s, selected: a.source }
    case 'reset':
      return empty
  }
}

const KEY = 'nyayagraph.session'
const RECENT = 'nyayagraph.recent'

function load(): Session {
  try {
    const raw = sessionStorage.getItem(KEY)
    if (!raw) return empty
    const s = JSON.parse(raw) as Session
    // an answer interrupted by a reload cannot resume
    return s.status === 'running' ? { ...s, status: 'stopped', stage: null } : s
  } catch {
    return empty
  }
}

export function recentQuestions(): string[] {
  try {
    return JSON.parse(localStorage.getItem(RECENT) ?? '[]')
  } catch {
    return []
  }
}

function remember(q: string) {
  try {
    const list = [q, ...recentQuestions().filter((x) => x !== q)].slice(0, 6)
    localStorage.setItem(RECENT, JSON.stringify(list))
  } catch {
    /* storage unavailable: recent list is a convenience */
  }
}

interface Store {
  session: Session
  run: (question: string, mode: Mode) => void
  stop: () => void
  select: (source: number | null) => void
  reset: () => void
}

const Ctx = createContext<Store | null>(null)

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, dispatch] = useReducer(reducer, undefined, load)
  const ctrl = useRef<AbortController | null>(null)

  useEffect(() => {
    if (session.status === 'running') return // save at milestones, not on every streamed word
    try {
      sessionStorage.setItem(KEY, JSON.stringify(session))
    } catch {
      /* ignore */
    }
  }, [session])

  const run = useCallback(async (question: string, mode: Mode) => {
    ctrl.current?.abort()
    const c = new AbortController()
    ctrl.current = c
    remember(question)
    dispatch({ type: 'start', question, mode })
    for await (const ev of ask(question, mode, c.signal)) {
      if (c.signal.aborted) return
      switch (ev.type) {
        case 'stage': dispatch({ type: 'stage', stage: ev.data.step }); break
        case 'context': dispatch({ type: 'context', context: ev.data }); break
        case 'text': dispatch({ type: 'text', block: ev.data.block, text: ev.data.text }); break
        case 'cite': dispatch({ type: 'cite', block: ev.data.block, source: ev.data.source }); break
        case 'done': dispatch({ type: 'done', done: ev.data }); return
        case 'error': dispatch({ type: 'error', error: ev.data }); return
      }
    }
    if (!c.signal.aborted) dispatch({ type: 'error', error: { code: 'stream_ended', message: 'The answer stopped before it finished.' } })
  }, [])

  const stop = useCallback(() => {
    ctrl.current?.abort()
    dispatch({ type: 'stopped' })
  }, [])

  const select = useCallback((source: number | null) => dispatch({ type: 'select', source }), [])
  const reset = useCallback(() => {
    ctrl.current?.abort()
    dispatch({ type: 'reset' })
  }, [])

  const value = useMemo(() => ({ session, run, stop, select, reset }), [session, run, stop, select, reset])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useSession(): Store {
  const s = useContext(Ctx)
  if (!s) throw new Error('useSession outside SessionProvider')
  return s
}
