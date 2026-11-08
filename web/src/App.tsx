import { useEffect } from 'react'
import { Link, Route, Routes, useLocation } from 'react-router-dom'
import CasePage from './pages/CasePage'
import NotFound from './pages/NotFound'
import ResearchPage from './pages/ResearchPage'
import { useSession } from './state'

export default function App() {
  const { pathname } = useLocation()
  const { reset, session } = useSession()

  useEffect(() => {
    if (!pathname.startsWith('/case/')) {
      document.title = session.question ? `${session.question} · Nyayagraph` : 'Nyayagraph'
    }
  }, [pathname, session.question])

  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="topbar">
        <div className="topbar-inner">
          <Link to="/" className="wordmark" onClick={() => pathname === '/' && reset()} aria-label="Nyayagraph, new question">
            <span className="wordmark-name" translate="no">Nyayagraph</span>
          </Link>
          <p className="topbar-scope">Supreme Court of India · criminal appeals</p>
        </div>
      </header>
      <main id="main" tabIndex={-1}>
        <Routes>
          <Route path="/" element={<ResearchPage />} />
          <Route path="/case/:id" element={<CasePage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </>
  )
}
