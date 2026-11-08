# Nyayagraph web app

React + TypeScript (Vite) front end for the GraphRAG research assistant. The FastAPI server in `api/server.py`
owns `/api` and, once built, serves this app too. Design context: `PRODUCT.md` at the repository root.

## Run

Prerequisites: Neo4j running with the graph loaded (`neo4j start`), `ANTHROPIC_API_KEY` in the root `.env`, the
retrieval index built (`python -m rag.embed`).

```bash
# one port: build once, then the API serves the site at http://localhost:8000
cd web && npm install && npm run build && cd ..
uvicorn api.server:app --port 8000
```

```bash
# development: API on 8000, Vite on 5173 with /api proxied
uvicorn api.server:app --port 8000
npm --prefix web run dev
```

## Structure

- `src/api.ts`: typed client; `/api/ask` is read as a server-sent event stream over `fetch`.
- `src/state.tsx`: the current question and answer, kept across navigation and in `sessionStorage`.
- `src/lib/text.ts`: answer rendering (Markdown with raw HTML disabled, citation markers, quotation marks on
  word-for-word passages), quote highlighting, citation formatting.
- `src/pages/ResearchPage.tsx`: question, streamed answer, trail of what was searched, sources panel (bottom sheet on
  phones). `src/pages/CasePage.tsx`: a judgment with provisions (and BNS / BNSS / BSA counterparts), citations in
  and out, numbered paragraphs; or a cited-only precedent with the judgments that cite it.
- `src/styles/tokens.css`: colour (OKLCH; light, and dark via the system setting), type and spacing tokens.
  Text pairs meet WCAG AA in both themes; control borders meet 3:1.
