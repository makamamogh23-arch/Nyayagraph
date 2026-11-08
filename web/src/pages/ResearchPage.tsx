import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { AnswerFoot, AnswerSkeleton, AnswerText } from '../components/Answer'
import AskBox from '../components/AskBox'
import Sources from '../components/Sources'
import { Progress, Trail } from '../components/Trail'
import { useMediaQuery, useMeta } from '../lib/hooks'
import { nf, numbering } from '../lib/text'
import { recentQuestions, useSession } from '../state'

export default function ResearchPage() {
  const { session, run, stop, select } = useSession()
  const { meta, failed } = useMeta()
  const narrow = useMediaQuery('(max-width: 899px)')
  const sheet = useRef<HTMLDialogElement>(null)
  const running = session.status === 'running'
  const [params, setParams] = useSearchParams()

  // the question lives in the URL so a link can be shared; opening one pre-fills it but never runs it (answers cost money)
  useEffect(() => {
    if (session.question && params.get('q') !== session.question) setParams({ q: session.question }, { replace: true })
  }, [session.question, params, setParams])

  if (session.status === 'idle') {
    return <Start onAsk={run} metaFailed={failed} meta={meta} initial={params.get('q') ?? ''} />
  }

  const ctx = session.context
  const sources = ctx?.sources ?? []
  const answering = running && session.stage === 'answering'
  const hasText = session.blocks.some((b) => b.text.trim())
  const citedCount = numbering(session.blocks).size

  const onCite = (i: number) => {
    select(i)
    if (narrow && sheet.current && !sheet.current.open) sheet.current.showModal()
  }

  return (
    <div className="research">
      <div className="research-ask">
        <AskBox key={`${session.question}|${session.mode}`} mode={session.mode} running={running} compact
          placeholder="Ask another question" onAsk={run} onStop={stop} />
      </div>

      <div className="research-grid">
        <article className="answer" aria-labelledby="answer-q" aria-busy={running}>
          <h1 id="answer-q" className="answer-q">{session.question}</h1>

          {running && session.stage && session.stage !== 'answering' && <Progress stage={session.stage} />}
          {ctx && <Trail context={ctx} passages={meta?.paragraphs ?? null} />}

          <div className="answer-body">
            {hasText ? (
              <AnswerText blocks={session.blocks} sources={sources} streaming={answering} selected={session.selected} onCite={onCite} />
            ) : running ? (
              <>
                {answering && <Progress stage="answering" />}
                <AnswerSkeleton />
              </>
            ) : null}

            {session.status === 'error' && session.error && (
              <Alert tone="danger" title={errorTitle(session.error.code)} action={
                <button type="button" className="btn btn--quiet btn--sm" onClick={() => run(session.question, session.mode)}>Try again</button>
              }>
                {session.error.message}
              </Alert>
            )}

            {session.status === 'stopped' && (
              <Alert tone="notice" title={hasText ? 'Stopped before the answer finished' : 'Stopped'} action={
                <button type="button" className="btn btn--quiet btn--sm" onClick={() => run(session.question, session.mode)}>Ask again</button>
              }>
                {hasText ? 'What is shown above is incomplete, and may lack citations for its last part.' : 'No answer was written.'}
              </Alert>
            )}

            {session.status === 'done' && session.done?.stop_reason === 'no_sources' && (
              <Alert tone="notice" title="Nothing in the collection matched">
                The {meta ? nf.format(meta.cases) : ''} judgments did not contain paragraphs close enough to this question.
                Try naming the provision (for example, “Section 438 CrPC”) or the doctrine in the words a judgment would use.
              </Alert>
            )}

            {session.status === 'done' && session.done && session.done.stop_reason !== 'no_sources' && (
              <AnswerFoot done={session.done} blocks={session.blocks} sources={sources} />
            )}
            <p className="visually-hidden" role="status">
              {session.status === 'done' && session.done?.stop_reason !== 'no_sources'
                ? `Answer ready. ${citedCount} ${citedCount === 1 ? 'source' : 'sources'} cited.`
                : ''}
            </p>
          </div>
        </article>

        {!narrow && ctx && (
          <aside className="panel" aria-label="Sources">
            <div className="panel-scroll">
              <Sources sources={sources} blocks={session.blocks} selected={session.selected} onSelect={select} answering={running} />
            </div>
          </aside>
        )}
      </div>

      {narrow && ctx && (
        <>
          <div className="sheet-bar">
            <button type="button" className="btn btn--primary sheet-open" onClick={() => sheet.current?.showModal()}>
              Sources <span className="num">· {citedCount ? `${citedCount} cited` : `${sources.length} read`}</span>
            </button>
          </div>
          <dialog ref={sheet} className="sheet" aria-label="Sources" onClick={(e) => e.target === sheet.current && sheet.current?.close()}>
            <div className="sheet-inner">
              <div className="sheet-head">
                <span className="sheet-grip" aria-hidden="true" />
                <button type="button" className="btn btn--quiet btn--sm" onClick={() => sheet.current?.close()}>Close</button>
              </div>
              <Sources sources={sources} blocks={session.blocks} selected={session.selected} onSelect={select} answering={running} />
            </div>
          </dialog>
        </>
      )}
    </div>
  )
}

function errorTitle(code: string): string {
  switch (code) {
    case 'no_api_key':
    case 'bad_api_key': return 'The answer service is not set up'
    case 'graph_unavailable': return 'The case graph is not running'
    case 'rate_limited': return 'Too many requests'
    case 'offline': return 'Cannot reach the server'
    default: return 'The answer could not be completed'
  }
}

function Alert({ tone, title, children, action }: { tone: 'danger' | 'notice'; title: string; children: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className={`alert alert--${tone}`} role={tone === 'danger' ? 'alert' : 'status'}>
      <p className="alert-title">{title}</p>
      <p className="alert-body">{children}</p>
      {action && <div className="alert-action">{action}</div>}
    </div>
  )
}

function Start({ onAsk, meta, metaFailed, initial }: {
  initial: string
  onAsk: (q: string, m: 'full' | 'quick') => void
  meta: ReturnType<typeof useMeta>['meta']
  metaFailed: boolean
}) {
  const [recent] = useState<string[]>(recentQuestions)
  const examples = (meta?.examples ?? []).slice(0, 6)

  return (
    <div className="start">
      <div className="start-intro">
        <h1 className="start-title">Research Indian criminal law from the judgments themselves</h1>
        <p className="start-lead">
          Ask a question. The answer cites the Supreme Court judgment and paragraph behind every statement, so you can read the
          source before you rely on it.
        </p>
      </div>

      <AskBox initial={initial} mode="full" running={false} autoFocus onAsk={onAsk} onStop={() => {}} />

      <p className="start-coverage num">
        {meta ? (
          <>
            {nf.format(meta.cases)} criminal-appeal judgments, {meta.years[0]}–{meta.years[1]} · {nf.format(meta.paragraphs)} paragraphs ·
            ask in IPC, CrPC and Evidence Act terms or their BNS, BNSS and BSA replacements
          </>
        ) : metaFailed ? (
          <span className="start-offline">The server is not responding. Start it with <code>uvicorn api.server:app</code>, then reload.</span>
        ) : (
          <span aria-hidden="true">&nbsp;</span>
        )}
      </p>

      <div className="start-lists">
        {recent.length > 0 && (
          <section aria-labelledby="recent-h">
            <h2 id="recent-h" className="list-title">Your recent questions</h2>
            <ul className="question-list">
              {recent.map((q) => (
                <li key={q}><button type="button" onClick={() => onAsk(q, 'full')}>{q}</button></li>
              ))}
            </ul>
          </section>
        )}
        {examples.length > 0 && (
          <section aria-labelledby="examples-h">
            <h2 id="examples-h" className="list-title">Questions to try</h2>
            <ul className="question-list">
              {examples.map((q) => (
                <li key={q}><button type="button" onClick={() => onAsk(q, 'full')}>{q}</button></li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  )
}
