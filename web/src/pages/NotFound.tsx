import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <div className="case case--message">
      <h1 className="case-title">Page not found</h1>
      <p className="case-message">
        There is nothing at this address. <Link to="/">Ask a question</Link> instead.
      </p>
    </div>
  )
}
