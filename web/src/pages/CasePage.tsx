import { useEffect, useMemo, useState } from 'react'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { getCase, HttpError, type AppealFrom, type CaseDetail, type CaseLink, type Counsel, type Provision, type Side, type Witness } from '../api'
import { useSession } from '../state'
import { caseHref, formatDate, formatDmy, nf, shortAct } from '../lib/text'

export default function CasePage() {
  const { id = '' } = useParams()
  const { hash } = useLocation()
  const navigate = useNavigate()
  const { session } = useSession()
  // keyed by id so a previous case never shows while the next one loads
  const [loaded, setLoaded] = useState<{ id: string; data?: CaseDetail; error?: { status: number; message: string } } | null>(null)
  const data = loaded?.id === id ? loaded.data ?? null : null
  const error = loaded?.id === id ? loaded.error ?? null : null

  useEffect(() => {
    const c = new AbortController()
    getCase(id, c.signal)
      .then((d) => setLoaded({ id, data: d }))
      .catch((e) => {
        if ((e as Error).name === 'AbortError') return
        setLoaded({ id, error: { status: e instanceof HttpError ? e.status : 0, message: e instanceof HttpError ? e.message : 'Could not reach the server.' } })
      })
    return () => c.abort()
  }, [id])

  useEffect(() => {
    if (data) document.title = `${data.title ?? data.label} · Nyayagraph`
  }, [data])

  // scroll to the requested paragraph once it is rendered
  useEffect(() => {
    if (!data || !hash) return
    const el = document.getElementById(decodeURIComponent(hash.slice(1)))
    if (el) {
      el.scrollIntoView({ block: 'start' })
      el.focus({ preventScroll: true })
    }
  }, [data, hash])

  const back = session.status !== 'idle' ? (
    <button type="button" className="backlink" onClick={() => navigate(-1)}>
      ← Back to the answer
    </button>
  ) : (
    <Link className="backlink" to="/">← New question</Link>
  )

  if (error) {
    return (
      <div className="case case--message">
        {back}
        <h1 className="case-title">{error.status === 404 ? 'Case not found' : 'This case could not be loaded'}</h1>
        <p className="case-message">{error.message}</p>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="case" aria-busy="true">
        {back}
        <div className="skeleton skeleton--title" aria-hidden="true"><span style={{ width: '60%' }} /></div>
        <div className="skeleton" aria-hidden="true"><span style={{ width: '40%' }} /><span className="skeleton-gap" /><span /><span /><span style={{ width: '80%' }} /></div>
        <span className="visually-hidden" role="status">Loading the judgment</span>
      </div>
    )
  }

  return data.in_corpus ? <Judgment data={data} hash={hash} back={back} /> : <Precedent data={data} back={back} />
}

function Judgment({ data, hash, back }: { data: CaseDetail; hash: string; back: React.ReactNode }) {
  const target = hash.slice(1)
  const date = formatDate(data.date)
  const judges = data.judges.authored.length ? data.judges.authored : data.judges.bench
  const multi = data.opinions.length > 1

  return (
    <div className="case">
      {back}
      <header className="case-head">
        <h1 className="case-title">{data.title ?? data.label}</h1>
        <dl className="case-facts">
          <div><dt>Neutral citation</dt><dd className="num">{data.case_id}</dd></div>
          {data.report && <div><dt>Reported</dt><dd className="num">{data.report}</dd></div>}
          {date && <div><dt>Decided</dt><dd>{date}</dd></div>}
          {judges.length > 0 && <div><dt>{data.judges.authored.length ? 'Author' : 'Bench'}</dt><dd>{judges.join(', ')}</dd></div>}
          {data.disposal && <div><dt>Outcome</dt><dd>{data.disposal}</dd></div>}
          {data.appeal_from?.[0] && <div><dt>Appeal from</dt><dd><AppealFromText a={data.appeal_from[0]} /></dd></div>}
        </dl>
      </header>

      <div className="case-grid">
        <section className="case-text" aria-label="Judgment text">
          {data.paragraphs.length === 0 ? (
            <p className="case-message">The text of this judgment is not in the collection.</p>
          ) : (
            <ol className="paras">
              {data.paragraphs.map((p, i) => {
                const pid = `p-${p.opinion}-${p.para_no}`
                const newOpinion = multi && (i === 0 || data.paragraphs[i - 1].opinion !== p.opinion)
                return (
                  <li key={p.pid} className="para-wrap">
                    {newOpinion && <h2 className="opinion-h">{p.opinion === 0 ? 'Judgment' : `Separate opinion ${p.opinion}`}</h2>}
                    <div id={pid} tabIndex={-1} className={`para${target === pid ? ' is-target' : ''}`}>
                      <a className="para-no num" href={`#${pid}`} aria-label={`Paragraph ${p.para_no}`}>{p.para_no}</a>
                      <p className="para-text prose">{p.text}</p>
                    </div>
                  </li>
                )
              })}
            </ol>
          )}
        </section>

        <aside className="case-side" aria-label="Provisions and citations">
          <Provisions items={data.provisions} />
          <CaseList title="Cites" empty="No citations to other judgments were found in the text." items={data.cites} anchorCase={data.case_id} showParas />
          <CaseList title="Cited by" empty="No later judgment in the collection cites this one." items={data.cited_by} showParas linkParas />
          {(data.witnesses?.length ?? 0) > 0 && <Witnesses items={data.witnesses!} caseId={data.case_id} />}
          {(data.counsel?.length ?? 0) > 0 && <CounselList items={data.counsel!} />}
        </aside>
      </div>
    </div>
  )
}

function Precedent({ data, back }: { data: CaseDetail; back: React.ReactNode }) {
  return (
    <div className="case">
      {back}
      <header className="case-head">
        <h1 className="case-title">{data.title ?? data.label}</h1>
        <dl className="case-facts">
          <div><dt>Citation</dt><dd className="num">{data.parallel.join(' · ') || data.label}</dd></div>
          {data.year && <div><dt>Year</dt><dd className="num">{data.year}</dd></div>}
        </dl>
        <p className="case-note">
          Not in the collection. Nyayagraph knows this judgment through the {nf.format(data.cited_by.length)}{' '}
          {data.cited_by.length === 1 ? 'judgment that cites' : 'judgments that cite'} it.
        </p>
      </header>
      <div className="case-solo">
        <CaseList title="Cited by" empty="No judgment in the collection cites it." items={data.cited_by} showParas linkParas initial={30} />
      </div>
    </div>
  )
}

function AppealFromText({ a }: { a: AppealFrom }) {
  const date = formatDmy(a.date)
  return (
    <>
      {a.court}
      {a.bench && <span className="fact-sub">, {a.bench} bench</span>}
      {(a.case_numbers.length > 0 || date) && (
        <span className="fact-sub num">
          {' · '}
          {a.case_numbers.slice(0, 2).join(', ')}
          {a.case_numbers.length > 2 && ` +${a.case_numbers.length - 2}`}
          {date && `${a.case_numbers.length ? ', ' : ''}decided ${date}`}
        </span>
      )}
    </>
  )
}

const SIDES: { key: Side; label: string }[] = [
  { key: 'appellant', label: 'For the appellant' },
  { key: 'respondent', label: 'For the respondent' },
  { key: 'parties', label: 'For the parties' },
  { key: 'complainant', label: 'For the complainant' },
  { key: 'intervenor', label: 'For the intervenor' },
  { key: 'amicus', label: 'Amicus curiae' },
  { key: 'other', label: 'Side not stated' },
]

const MODEL_NOTE = 'Extracted from the headnote by the fine-tuned Qwen2.5-7B NER model'

function CounselList({ items }: { items: Counsel[] }) {
  return (
    <section className="side-block" aria-labelledby="counsel-h">
      <h2 id="counsel-h" className="side-title">Counsel <span className="side-count num">{items.length}</span></h2>
      <p className="side-note">{MODEL_NOTE}.</p>
      {SIDES.map(({ key, label }) => {
        const group = items.filter((c) => c.side === key)
        if (!group.length) return null
        return (
          <div key={key} className="counsel-group">
            <h3 className="counsel-side">{label}</h3>
            <ul className="counsel-list">
              {group.map((c) => (
                <li key={c.name}>
                  {c.name}
                  {c.senior && <abbr className="counsel-sr" title="Senior Advocate">Sr.</abbr>}
                  {c.cases > 1 && <span className="counsel-n num"> · {c.cases} cases</span>}
                </li>
              ))}
            </ul>
          </div>
        )
      })}
    </section>
  )
}

function Witnesses({ items, caseId }: { items: Witness[]; caseId: string }) {
  return (
    <section className="side-block" aria-labelledby="wit-h">
      <h2 id="wit-h" className="side-title">Prosecution witnesses <span className="side-count num">{items.length}</span></h2>
      <p className="side-note">Extracted from the judgment text by the fine-tuned Qwen2.5-7B NER model.</p>
      <ul className="witness-list">
        {items.map((w) => (
          <li key={w.pw}>
            <span className="witness-pw num">PW-{w.pw}</span>
            <span className="witness-name">{w.name}</span>
            <span className="case-link-meta num"><ParaLinks caseId={caseId} paras={w.paras} inPage /></span>
          </li>
        ))}
      </ul>
    </section>
  )
}

function Provisions({ items }: { items: Provision[] }) {
  const [all, setAll] = useState(false)
  const shown = all ? items : items.slice(0, 12)
  return (
    <section className="side-block" aria-labelledby="prov-h">
      <h2 id="prov-h" className="side-title">Provisions applied <span className="side-count num">{items.length}</span></h2>
      {items.length === 0 ? (
        <p className="side-empty">No statutory provisions were identified.</p>
      ) : (
        <ul className="prov-list">
          {shown.map((p) => (
            <li key={p.act + p.section}>
              <span className="prov-name">{p.act === 'Constitution of India' ? `Article ${p.section}` : `${shortAct(p.act)} s.${p.section}`}</span>
              {p.maps_to.length > 0 && (
                <span className="prov-new"> = {p.maps_to.map((m) => `${shortAct(m.act)} s.${m.section}${m.sub && !m.sub.startsWith('(') ? ' ' : ''}${m.sub}`).join(', ')}</span>
              )}
              <span className="prov-paras num">{paraList(p.paras)}</span>
            </li>
          ))}
        </ul>
      )}
      {items.length > 12 && (
        <button type="button" className="link-btn" onClick={() => setAll((v) => !v)}>{all ? 'Show fewer' : `Show all ${items.length}`}</button>
      )}
    </section>
  )
}

function CaseList({ title, items, empty, showParas, linkParas, anchorCase, initial = 10 }: {
  title: string; items: CaseLink[]; empty: string; showParas?: boolean; linkParas?: boolean; anchorCase?: string; initial?: number
}) {
  const [all, setAll] = useState(false)
  const shown = all ? items : items.slice(0, initial)
  const hid = useMemo(() => `h-${title.replace(/\W+/g, '-').toLowerCase()}`, [title])
  return (
    <section className="side-block" aria-labelledby={hid}>
      <h2 id={hid} className="side-title">{title} <span className="side-count num">{items.length}</span></h2>
      {items.length === 0 ? (
        <p className="side-empty">{empty}</p>
      ) : (
        <ul className="case-links">
          {shown.map((c) => (
            <li key={c.case_id}>
              <Link to={caseHref(c.case_id)} className="case-link">{c.title ?? c.label}</Link>
              <span className="case-link-meta num">
                {c.case_id.startsWith('ext:') ? c.label : c.case_id}
                {showParas && c.paras.length > 0 && (
                  <>
                    {' · '}
                    {linkParas ? (
                      <ParaLinks caseId={c.case_id} paras={c.paras} />
                    ) : anchorCase ? (
                      <ParaLinks caseId={anchorCase} paras={c.paras} inPage />
                    ) : paraList(c.paras)}
                  </>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
      {items.length > initial && (
        <button type="button" className="link-btn" onClick={() => setAll((v) => !v)}>{all ? 'Show fewer' : `Show all ${nf.format(items.length)}`}</button>
      )}
    </section>
  )
}

function ParaLinks({ caseId, paras, inPage }: { caseId: string; paras: string[]; inPage?: boolean }) {
  const list = paras.slice(0, 4)
  return (
    <>
      ¶{' '}
      {list.map((p, i) => {
        const [op, no] = p.split(':')
        const href = inPage ? `#p-${op}-${no}` : caseHref(caseId, Number(op), Number(no))
        return (
          <span key={p}>
            {i > 0 && ', '}
            {inPage ? <a href={href}>{no}</a> : <Link to={href}>{no}</Link>}
          </span>
        )
      })}
      {paras.length > 4 && ` +${paras.length - 4}`}
    </>
  )
}

function paraList(paras: string[]): string {
  const nums = paras.map((p) => p.split(':')[1])
  return `¶ ${nums.slice(0, 5).join(', ')}${nums.length > 5 ? ` +${nums.length - 5}` : ''}`
}
