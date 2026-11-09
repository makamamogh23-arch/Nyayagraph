import { Link } from 'react-router-dom'
import type { Context, Stage } from '../api'
import { caseHref, nf, parseCorrespondence, provisionLabel } from '../lib/text'

const STEPS: { key: Stage; label: string }[] = [
  { key: 'analysing', label: 'Reading the question' },
  { key: 'retrieving', label: 'Searching the judgments' },
  { key: 'answering', label: 'Writing the answer' },
]

export function Progress({ stage }: { stage: Stage }) {
  const at = STEPS.findIndex((s) => s.key === stage)
  return (
    <ol className="progress" aria-label="Progress">
      {STEPS.map((s, i) => {
        const state = i < at ? 'done' : i === at ? 'active' : 'pending'
        return (
          <li key={s.key} className={`progress-step is-${state}`} aria-current={state === 'active' ? 'step' : undefined}>
            <span className="progress-dot" aria-hidden="true" />
            {s.label}
            {state === 'done' && <span className="visually-hidden"> (done)</span>}
          </li>
        )
      })}
    </ol>
  )
}

/** What the system did before answering: provisions and their old/new counterparts, named cases, scope, how much it read. */
export function Trail({ context, passages }: { context: Context; passages: number | null }) {
  const corr = context.provision_map.map(parseCorrespondence).filter(Boolean) as { old: string; next: string }[]
  const mapped = corr.flatMap((c) => [c.old, c.next])
  // 'BSA s.23' is already shown inside 'Evidence Act s.27 = BSA s.23 proviso'
  const plain = context.provisions.map(provisionLabel).filter((p) => !mapped.some((m) => m === p || m.startsWith(`${p}(`) || m.startsWith(`${p} `)))
  const defs = (context.related_map ?? []).map(parseCorrespondence).filter(Boolean) as { old: string; next: string; label?: string }[]
  const judgments = new Set(context.sources.map((s) => s.case_id)).size
  const precedents = context.sources.filter((s) => s.precedent).length

  return (
    <dl className="trail" aria-label="How this was answered">
      {(corr.length > 0 || plain.length > 0) && (
        <div className="trail-row">
          <dt>Provisions</dt>
          <dd className="trail-chips">
            {corr.map((c) => (
              <span key={c.old + c.next} className="chip" title="Correspondence from the official comparison tables">
                {c.old} <span className="chip-eq" aria-label="corresponds to">=</span> {c.next}
              </span>
            ))}
            {plain.map((p) => (
              <span key={p} className="chip">{p}</span>
            ))}
          </dd>
        </div>
      )}
      {defs.length > 0 && (
        <div className="trail-row">
          <dt>Defined in</dt>
          <dd className="trail-chips">
            {defs.map((c) => (
              <span key={c.old + c.next} className="chip" title="The section defining the offence, from the same comparison tables">
                {c.label && <span className="chip-note">{c.label}: </span>}
                {c.old} <span className="chip-eq" aria-label="corresponds to">=</span> {c.next}
              </span>
            ))}
          </dd>
        </div>
      )}
      {context.named_cases.length > 0 && (
        <div className="trail-row">
          <dt>Cases named</dt>
          <dd className="trail-chips">
            {context.named_cases.map((c) => (
              <Link key={c.case_id} className="chip chip--link" to={caseHref(c.case_id)}>
                {c.title ?? c.label}
                {c.case_id.startsWith('ext:') ? <span className="chip-note"> · {c.label}</span> : null}
              </Link>
            ))}
          </dd>
        </div>
      )}
      {(context.scope?.length ?? 0) > 0 && (
        <div className="trail-row">
          <dt>Limited to</dt>
          <dd className="trail-chips">
            {context.scope!.map((f) => (
              <span key={f.kind + f.label} className="chip" title="From the case graph (automated extraction of the judgments' headnotes and text)">
                <span className="chip-note">{f.kind === 'witness' ? `${f.detail}: ` : `${f.detail} `}</span>
                {f.label}
                {f.kind !== 'witness' && <span className="chip-note"> · {nf.format(f.cases)} {f.cases === 1 ? 'judgment' : 'judgments'}</span>}
              </span>
            ))}
          </dd>
        </div>
      )}
      <div className="trail-row">
        <dt>Read</dt>
        <dd className="num">
          {context.sources.length} paragraphs from {judgments} {judgments === 1 ? 'judgment' : 'judgments'}
          {precedents > 0 && <>, including {precedents} from cited precedents</>}
          {passages && !context.scope?.length ? <span className="trail-muted"> · searched {nf.format(passages)} paragraphs</span> : null}
        </dd>
      </div>
    </dl>
  )
}
