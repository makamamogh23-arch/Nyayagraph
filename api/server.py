"""Web API for the Nyayagraph research assistant (serves the React app in web/dist when built).

  GET  /api/meta                 corpus coverage + example questions
  POST /api/ask                  {question, mode: "full" | "quick"} -> text/event-stream:
                                   stage   {step}                          analysing | retrieving | answering
                                   context {provisions, provision_map, named_cases, sources}
                                   text    {block, text}                   answer text, by content block
                                   cite    {block, source, cited_text}     a citation on that block
                                   done    {model, usage, cost_usd, seconds, stop_reason}
                                   error   {code, message}
  GET  /api/case/{case_id}       a judgment (corpus) or a cited-only precedent (external): metadata, judges,
                                 provisions with old/new counterparts, cites, cited by, paragraphs
  GET  /api/paragraph/{pid}      one paragraph

Run: uvicorn api.server:app --port 8000   (dev: npm run dev in web/, which proxies /api here)
"""
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Neo4j logs a warning for every query naming a relationship type not loaded yet (e.g. WITNESS_IN before the
# witness pass): those queries are expected to return nothing
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

from rag.ask import PRICES, build_request, load_key, log_answer

MODES = {"full": ("claude-opus-5-5", "medium"), "quick": ("claude-haiku-5-5", "low")}
DIST = "web/dist"



@asynccontextmanager
async def lifespan(_app):
    """Load the search index in the background so the first question doesn't pay for it."""
    def go():
        try:
            tables()
            retriever()
        except Exception:
            pass  # reported on first use (e.g. Neo4j not running)
    threading.Thread(target=go, daemon=True).start()
    yield


app = FastAPI(title="Nyayagraph", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)


# --- display helpers --------------------------------------------------------------------------------------------

SMALL = {"of", "and", "the", "by", "through", "in", "for", "to", "at", "on", "or", "via", "with"}
KEEP = {"ors.": "Ors.", "ors": "Ors.", "anr.": "Anr.", "anr": "Anr.", "etc.": "etc.", "etc": "etc.", "v.": "v.",
        "nct": "NCT", "cbi": "CBI", "nia": "NIA", "uoi": "UOI", "ed": "ED", "lrs.": "LRs.", "lrs": "LRs.",
        "pvt.": "Pvt.", "ltd.": "Ltd.", "dr.": "Dr.", "mr.": "Mr.", "smt.": "Smt.", "sh.": "Sh."}


def display_title(t: str | None) -> str | None:
    """'UDAYAKUMAR versus STATE OF TAMIL NADU' -> 'Udayakumar v. State of Tamil Nadu'; mixed-case titles keep
    their casing (only 'versus' / 'vs.' become 'v.')."""
    if not isinstance(t, str) or not t.strip():
        return None
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+(?:versus|vs\.?|v/s\.?)\s+", " v. ", t, flags=re.I)
    letters = [c for c in t if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.7:
        return t
    def cap(w: str) -> str:  # first letter up, after any leading "(" or quote
        m = re.match(r"^([^A-Za-z]*)(.*)$", w)
        return m[1] + m[2][:1].upper() + m[2][1:].lower()

    out = []
    for i, w in enumerate(t.split(" ")):
        lw = w.lower().strip("(),")
        starts = i == 0 or out[-1] == "v."  # each party name starts capitalised
        if lw in KEEP:
            out.append(w.lower().replace(lw, KEEP[lw]))
        elif re.fullmatch(r"\(?(?:[A-Za-z]\.){1,4}[A-Za-z]?\.?,?\)?", w):  # U.P., M.P., K.A.
            out.append(w.upper())
        elif not starts and lw in SMALL:
            out.append(w.lower())
        else:
            out.append("-".join(cap(p) for p in w.split("-")))
    return " ".join(out)


def display_key(key: str) -> str:
    """'SCC|2013|7|211' -> '(2013) 7 SCC 211'; 'AIR SC|1977||45' -> 'AIR 1977 SC 45'; 'SCR|1962|3|10' ->
    '[1962] 3 SCR 10'; 'SCC OnLine SC|2019||1346' -> '2019 SCC OnLine SC 1346'."""
    parts = key.split("|")
    if len(parts) != 4:
        return key
    rep, year, vol, page = parts
    if rep.startswith("AIR"):
        court = rep[3:].strip()
        return f"AIR {year} {court} {page}".replace("  ", " ")
    if rep.startswith("SCC OnLine") or rep == "INSC":
        return f"{year} {rep} {page}"
    if rep.startswith("SCC"):
        return f"({year}) {vol} {rep} {page}".replace("  ", " ")
    if rep.startswith("SCR"):
        return f"[{year}] {vol} {rep} {page}".replace("  ", " ")
    return f"{year} {rep} {vol} {page}".replace("  ", " ")


def case_label(case_id: str, keys=None) -> str:
    if not case_id.startswith("ext:"):
        return case_id
    keys = list(keys) if keys is not None else [case_id[4:]]
    return display_key(keys[0]) if keys else case_id[4:]


# --- state ------------------------------------------------------------------------------------------------------

_lock = threading.Lock()
_retriever = None


def retriever():
    """Loaded on first use (about 20 s): BM25 index, embeddings, Neo4j connection."""
    global _retriever
    with _lock:
        if _retriever is None:
            from rag.retrieve import Retriever
            _retriever = Retriever()
            _ = _retriever.bm25, _retriever.encoder
        return _retriever


@lru_cache(maxsize=1)
def tables() -> dict:
    cases = pd.read_parquet("data/graph/nodes_case.parquet")
    paras = pd.read_parquet("data/graph/nodes_paragraph.parquet")
    paras = paras.sort_values(["case_id", "opinion", "para_no"])
    by_case = {cid: g for cid, g in paras.groupby("case_id", sort=False)}
    return {"cases": cases.set_index("case_id"), "paras_by_case": by_case,
            "para": dict(zip(paras["pid"], zip(paras["case_id"], paras["opinion"], paras["para_no"], paras["text"])))}


def neo4j():
    from graph.conn import driver
    return driver()


# --- endpoints --------------------------------------------------------------------------------------------------

def example_questions() -> list[str]:
    try:
        return [l.strip() for l in open("rag/questions.txt") if l.strip() and not l.startswith("#")]
    except OSError:
        return []


@app.get("/api/meta")
def meta():
    t = tables()
    c = t["cases"]
    corpus = c[c["in_corpus"]]
    return {"cases": int(len(corpus)), "paragraphs": int(sum(len(g) for g in t["paras_by_case"].values())),
            "precedents": int((~c["in_corpus"]).sum()), "years": [int(corpus["year"].min()), int(corpus["year"].max())],
            "court": "Supreme Court of India", "examples": example_questions(),
            "modes": {k: {"model": m, "price_per_mtok": PRICES.get(m)} for k, (m, _) in MODES.items()}}


class AskBody(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: str = Field(default="full", pattern="^(full|quick)$")


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def source_json(i: int, h) -> dict:
    c = tables()["cases"]
    row = c.loc[h.case_id] if h.case_id in c.index else None
    return {"index": i, "pid": h.pid, "label": h.label, "case_id": h.case_id, "opinion": h.opinion,
            "para_no": h.para_no, "title": display_title(h.title) or h.case_id, "year": h.year,
            "report": (row["citation"] if row is not None and isinstance(row["citation"], str) else None),
            "text": h.text, "why": h.why, "precedent": bool(h.cited_from), "cited_from": h.cited_from}


def ask_stream(question: str, mode: str):
    import anthropic

    model, effort = MODES[mode]
    t0 = time.time()
    yield sse("stage", {"step": "analysing"})
    load_key()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        yield sse("error", {"code": "no_api_key", "message": "Add ANTHROPIC_API_KEY to the project's .env file, then "
                                                            "restart the server."})
        return
    try:
        r = retriever()
        yield sse("stage", {"step": "retrieving"})
        req = build_request(question, r, model, effort)
    except Exception as e:  # Neo4j down, index files missing
        msg = str(e)
        code = "graph_unavailable" if "7687" in msg or "Neo4j" in msg or "ServiceUnavailable" in type(e).__name__ \
            else "retrieval_failed"
        yield sse("error", {"code": code, "message": "Start Neo4j with `neo4j start`, wait a few seconds, then "
                                                     "try again." if code == "graph_unavailable" else
                            f"Retrieval failed: {msg[:300]}"})
        return
    search, hits = req["search"], req["hits"]
    named = [{"case_id": cid, "label": case_label(cid), "title": display_title(r.all_titles.get(cid))}
             for cid in search["named_cases"]]
    yield sse("context", {"provisions": [p.replace("|", " s.") for p in search["provisions"]],
                          "provision_map": search["provision_map"], "related_map": search.get("related_map", []),
                          "named_cases": named,
                          "sources": [source_json(i, h) for i, h in enumerate(hits)],
                          "seconds": round(time.time() - t0, 1)})
    if req["params"] is None:
        yield sse("done", {"model": model, "usage": None, "cost_usd": 0, "seconds": round(time.time() - t0, 1),
                           "stop_reason": "no_sources"})
        return

    yield sse("stage", {"step": "answering"})
    client = anthropic.Anthropic()
    block_map: dict[int, int] = {}  # API content-block index -> our text-block number
    blocks: list[dict] = []
    try:
        with client.messages.stream(**req["params"]) as stream:
            for ev in stream:
                if ev.type == "content_block_start" and ev.content_block.type == "text":
                    block_map[ev.index] = len(blocks)
                    blocks.append({"text": "", "cites": []})
                elif ev.type == "content_block_delta" and ev.index in block_map:
                    b = block_map[ev.index]
                    if ev.delta.type == "text_delta":
                        blocks[b]["text"] += ev.delta.text
                        yield sse("text", {"block": b, "text": ev.delta.text})
                    elif ev.delta.type == "citations_delta" and ev.delta.citation.type == "search_result_location":
                        c = ev.delta.citation
                        blocks[b]["cites"].append(c.search_result_index)
                        yield sse("cite", {"block": b, "source": c.search_result_index, "cited_text": c.cited_text})
            msg = stream.get_final_message()
    except anthropic.AuthenticationError:
        yield sse("error", {"code": "bad_api_key", "message": "The Anthropic API key was rejected. Check "
                                                              "ANTHROPIC_API_KEY in .env."})
        return
    except anthropic.RateLimitError:
        yield sse("error", {"code": "rate_limited", "message": "The model is rate-limited right now. Wait a minute "
                                                               "and ask again."})
        return
    except anthropic.APIError as e:
        yield sse("error", {"code": "model_error", "message": f"The model request failed: {str(e)[:200]}"})
        return

    u = msg.usage
    p_in, p_out = PRICES.get(model, (0, 0))
    cost = (u.input_tokens * p_in + u.output_tokens * p_out) / 1e6
    seconds = round(time.time() - t0, 1)
    yield sse("done", {"model": model, "usage": {"in": u.input_tokens, "out": u.output_tokens},
                       "cost_usd": round(cost, 5), "seconds": seconds, "stop_reason": msg.stop_reason})
    cited = sorted({i for b in blocks for i in b["cites"]})
    log_answer({"question": question, "model": model, "via": "web", "blocks": blocks,
                "sources": [{"label": hits[i].label, "case_id": hits[i].case_id, "para_no": hits[i].para_no}
                            for i in cited],
                "retrieved": [h.label for h in hits], "provisions": search["provisions"],
                "provision_map": search["provision_map"], "named_cases": search["named_cases"],
                "stop_reason": msg.stop_reason, "usage": {"in": u.input_tokens, "out": u.output_tokens},
                "cost_usd": round(cost, 5), "seconds": {"total": seconds}})


@app.post("/api/ask")
def ask(body: AskBody):
    return StreamingResponse(ask_stream(body.question.strip(), body.mode), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/paragraph/{pid:path}")
def paragraph(pid: str):
    p = tables()["para"].get(pid)
    if p is None:
        raise HTTPException(404, "Paragraph not found")
    case_id, opinion, para_no, text = p
    return {"pid": pid, "case_id": case_id, "opinion": int(opinion), "para_no": int(para_no), "text": text}


@app.get("/api/case/{case_id:path}")
def case(case_id: str):
    t = tables()
    if case_id not in t["cases"].index:
        raise HTTPException(404, "No case with that citation in the collection")
    row = t["cases"].loc[case_id]
    keys = list(row["keys"]) if row["keys"] is not None else []
    in_corpus = bool(row["in_corpus"])
    out = {"case_id": case_id, "in_corpus": in_corpus, "label": case_label(case_id, keys),
           "title": display_title(row["title"]), "date": row["decision_date"] if isinstance(row["decision_date"], str) else None,
           "year": None if pd.isna(row["year"]) else int(row["year"]),
           "report": row["citation"] if isinstance(row["citation"], str) else None,
           "disposal": re.sub(r"^Case\s+", "", row["disposal"]).capitalize() if isinstance(row["disposal"], str) else None,
           "parallel": [display_key(k) for k in keys]}
    try:
        with neo4j() as d, d.session() as s:
            out["judges"] = s.run("""MATCH (c:Case {case_id: $id})
                                     OPTIONAL MATCH (a:Judge)-[:AUTHORED]->(c)
                                     OPTIONAL MATCH (b:Judge)-[:SAT_ON]->(c)
                                     RETURN collect(DISTINCT a.name) AS authored, collect(DISTINCT b.name) AS bench""",
                                  id=case_id).single().data()
            out["provisions"] = s.run("""MATCH (c:Case {case_id: $id})-[e:INVOKES]->(p:Provision)
                                         OPTIONAL MATCH (p)-[m:MAPS_TO]->(n:Provision)
                                         RETURN p.act AS act, p.section AS section, e.paras AS paras, e.n AS n,
                                                collect(CASE WHEN n IS NULL THEN NULL ELSE
                                                  {act: n.act, section: n.section, sub: m.new_sub} END) AS maps_to
                                         ORDER BY n DESC, act, section""", id=case_id).data()
            out["cites"] = s.run("""MATCH (c:Case {case_id: $id})-[e:CITES]->(d:Case)
                                    RETURN d.case_id AS case_id, d.title AS title, d.year AS year, d.keys AS keys,
                                           d.in_corpus AS in_corpus, e.paras AS paras
                                    ORDER BY size(e.paras) DESC, d.year DESC""", id=case_id).data()
            # from the fine-tuned NER pass (ner/corpus_entities.py); empty when that pass has not been loaded
            out["appeal_from"] = s.run("""MATCH (c:Case {case_id: $id})-[e:APPEAL_FROM]->(k:Court)
                                          RETURN k.name AS court, e.bench AS bench, e.date AS date,
                                                 e.case_numbers AS case_numbers, e.source AS source""",
                                       id=case_id).data()
            out["counsel"] = s.run("""MATCH (l:Lawyer)-[e:APPEARED_IN]->(c:Case {case_id: $id})
                                      OPTIONAL MATCH (l)-[o:APPEARED_IN]->(:Case)
                                      RETURN l.name AS name, e.side AS side, e.senior AS senior, count(o) AS cases
                                      ORDER BY e.side, e.senior DESC, cases DESC, name""", id=case_id).data()
            out["witnesses"] = s.run("""MATCH (w:Witness)-[e:WITNESS_IN]->(c:Case {case_id: $id})
                                        RETURN w.name AS name, w.pw AS pw, e.paras AS paras ORDER BY w.pw""",
                                     id=case_id).data()
            out["cited_by"] = s.run("""MATCH (s:Case)-[e:CITES]->(c:Case {case_id: $id})
                                       RETURN s.case_id AS case_id, s.title AS title, s.year AS year, e.paras AS paras
                                       ORDER BY s.year DESC, s.case_id""", id=case_id).data()
    except Exception:
        raise HTTPException(503, "The case graph is not reachable. Start Neo4j (neo4j start) and reload.")
    for k in ("cites", "cited_by"):
        for x in out[k]:
            x["title"] = display_title(x["title"])
            x["label"] = case_label(x["case_id"], x.pop("keys", None))
    g = t["paras_by_case"].get(case_id)
    out["paragraphs"] = [] if g is None else [
        {"pid": p.pid, "opinion": int(p.opinion), "para_no": int(p.para_no), "text": p.text} for p in g.itertuples()]
    out["opinions"] = sorted({p["opinion"] for p in out["paragraphs"]})
    return out


# --- the built web app -------------------------------------------------------------------------------------------

if os.path.isdir(DIST):
    app.mount("/assets", StaticFiles(directory=f"{DIST}/assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = os.path.join(DIST, path)
        return FileResponse(f if path and os.path.isfile(f) else f"{DIST}/index.html")
