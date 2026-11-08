"""GraphRAG research assistant: retrieve (rag/retrieve.py), then answer with Claude using search_result blocks,
so every claim carries a native citation to a retrieved paragraph (case_id + para_no).

Each Q&A (question, retrieved labels, answer, citations, usage) is appended to data/rag/answers.jsonl.

Usage:
  python -m rag.ask "When can a conviction rest solely on circumstantial evidence?"
  python -m rag.ask --model claude-sonnet-5-5 "..."
  python -m rag.ask --show-context "..."
"""
import argparse
import re
import json
import os
import time

from rag.retrieve import Hit, Retriever

MODEL = "claude-opus-5-5"
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}
LOG = "data/rag/answers.jsonl"

SYSTEM = """You are a legal research assistant for Indian criminal law, working from paragraphs of Supreme Court of \
India judgments supplied as search results. Each result's source is "<case id> ¶<paragraph>"; its title names the case.

Rules:
- Answer only from the supplied paragraphs. Every legal proposition, test, holding or fact must be supported by a \
citation to the paragraph it comes from. If the paragraphs do not answer the question, or only partly, say so plainly \
and state what is missing; do not fill gaps from general knowledge.
- Distinguish what the Court held from what it merely recorded: counsel's submissions, the lower court's view, or a \
passage the Court quotes in order to reject it. Say which when it matters.
- Name the case when you rely on it. Where judgments differ or develop a rule over time, say so and give each.
- Provisions: the IPC, CrPC and Evidence Act were replaced from 1 July 2024 by the BNS, BNSS and BSA. Give the \
counterpart of a provision when the correspondence note under the question or the paragraphs supply it, and say \
it comes from the comparison table; otherwise do not guess section numbers. Most judgments here predate the new \
Acts and apply the old sections.
- Write for a lawyer: lead with the direct answer, then the supporting authorities. Plain prose, short paragraphs; \
no headings for short answers."""


def search_result(h: Hit) -> dict:
    title = f"{h.title} ({h.year})" if h.year else h.title
    if h.cited_from:
        title += f" — precedent cited at {h.cited_from}"
    return {"type": "search_result", "source": h.label, "title": title or h.label,
            "content": [{"type": "text", "text": h.text}], "citations": {"enabled": True}}


def load_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY") or not os.path.exists(".env"):
        return
    for line in open(".env"):
        k, _, v = line.strip().removeprefix("export ").partition("=")
        if k.strip() == "ANTHROPIC_API_KEY" and v.strip():
            os.environ["ANTHROPIC_API_KEY"] = v.strip().strip("'\"")


def render(message, hits: list[Hit]) -> tuple[str, list[dict]]:
    """Answer text with [n] markers after cited spans + the numbered source list actually cited."""
    order: dict[int, int] = {}
    out = []
    def squash(t: str) -> str:
        return re.sub(r"\W+", " ", t).lower().strip()

    for block in message.content:
        if block.type != "text":
            continue
        text = block.text
        cited = " ".join(c.cited_text for c in block.citations or [] if c.type == "search_result_location")
        core = text.strip()
        # a cited span that appears word for word in its paragraph is a quotation: mark it as one
        if cited and len(core) > 20 and squash(core) in squash(cited) and not core.startswith(("\"", "“")):
            lead, trail = text[:len(text) - len(text.lstrip())], text[len(text.rstrip()):]
            text = f"{lead}“{core}”{trail}"
        out.append(text)
        marks = []
        for c in block.citations or []:
            if c.type != "search_result_location":
                continue
            n = order.setdefault(c.search_result_index, len(order) + 1)
            if n not in marks:
                marks.append(n)
        if marks:
            out.append("".join(f"[{n}]" for n in sorted(marks)))
    sources = []
    for idx, n in sorted(order.items(), key=lambda x: x[1]):
        h = hits[idx]
        sources.append({"n": n, "label": h.label, "case_id": h.case_id, "para_no": h.para_no,
                        "opinion": h.opinion, "title": h.title, "year": h.year, "precedent": bool(h.cited_from)})
    return "".join(out).strip(), sources


def build_request(question: str, retriever: Retriever, model: str = MODEL, effort: str = "medium") -> dict:
    """Retrieve, then build the Messages request. Shared by ask() and the web API (api/server.py).
    Returns {search, hits, params}; params is None when nothing relevant was retrieved."""
    r = retriever.search(question)
    hits = r["hits"] + r["precedents"]
    if not hits:
        return {"search": r, "hits": hits, "params": None}
    content = [search_result(h) for h in hits]
    note = ""
    if r["provision_map"]:
        note = ("\n\nOld/new correspondence for provisions named in the question (from the official comparison "
                "tables in the graph): " + "; ".join(r["provision_map"]))
    elif r["provisions"]:
        note = "\n\nProvisions named in the question: " + "; ".join(p.replace("|", " s.") for p in r["provisions"])
    content.append({"type": "text", "text": question + note})
    params = dict(model=model, max_tokens=8000, system=SYSTEM, thinking={"type": "adaptive"},
                  output_config={"effort": effort}, messages=[{"role": "user", "content": content}])
    return {"search": r, "hits": hits, "params": params}


def log_answer(rec: dict) -> None:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def ask(question: str, retriever: Retriever, model: str = MODEL, effort: str = "medium") -> dict:
    import anthropic

    load_key()
    t0 = time.time()
    req = build_request(question, retriever, model, effort)
    r, hits = req["search"], req["hits"]
    t_retrieve = time.time() - t0
    if req["params"] is None:
        return {"question": question, "answer": "No relevant paragraphs found in the corpus.", "sources": []}

    client = anthropic.Anthropic()
    with client.messages.stream(**req["params"]) as stream:
        msg = stream.get_final_message()
    answer, sources = render(msg, hits)
    p_in, p_out = PRICES.get(model, (0, 0))
    u = msg.usage
    cost = (u.input_tokens * p_in + u.output_tokens * p_out) / 1e6
    blocks = [{"text": b.text, "cites": [c.search_result_index for c in (b.citations or [])
                                         if c.type == "search_result_location"]}
              for b in msg.content if b.type == "text"]
    rec = {"question": question, "model": model, "answer": answer, "sources": sources, "blocks": blocks,
           "retrieved": [h.label for h in hits], "provisions": r["provisions"], "provision_map": r["provision_map"], "named_cases": r["named_cases"],
           "stop_reason": msg.stop_reason, "usage": {"in": u.input_tokens, "out": u.output_tokens},
           "cost_usd": round(cost, 5), "seconds": {"retrieve": round(t_retrieve, 1), "total": round(time.time() - t0, 1)}}
    log_answer(rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()
    retriever = Retriever()
    q = " ".join(args.question)
    if args.show_context:
        r = retriever.search(q)
        for h in r["hits"] + r["precedents"]:
            print(f"--- {h.label}  {h.title}  [{', '.join(h.why)}{'; cited from ' + h.cited_from if h.cited_from else ''}]")
            print(h.text[:600], "\n")
    rec = ask(q, retriever, args.model, args.effort)
    print(rec["answer"], "\n")
    for s in rec["sources"]:
        tag = " (precedent)" if s["precedent"] else ""
        print(f"[{s['n']}] {s['title']} ({s['year']}) — {s['label']}{tag}")
    if "usage" in rec:
        print(f"\n{rec['model']}  in {rec['usage']['in']:,} / out {rec['usage']['out']:,} tokens  ${rec['cost_usd']:.4f}  "
              f"{rec['seconds']['total']}s  stop={rec['stop_reason']}")


if __name__ == "__main__":
    main()
