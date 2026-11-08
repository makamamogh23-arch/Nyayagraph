"""Free, deterministic checks on logged answers (data/rag/answers.jsonl, records that kept their raw blocks).
No model calls.

  quotes    every quoted span ("..." or “...”, at least 4 words) in a cited passage must appear word for word in a
            paragraph that passage cites (ellipses split a quote into pieces, each checked). Outcomes:
              exact         found in a cited paragraph
              other_source  not in the cited paragraph(s), but in another paragraph the answer had
              not_found     in no retrieved paragraph (a paraphrase in quotation marks, or an OCR-corrected quote)
            Quoted spans in uncited passages are checked against all retrieved paragraphs.
  coverage  share of the answer's prose (characters, headings and blank lines excluded) inside cited passages.
  sections  section / article numbers the answer states that appear nowhere in the question, the correspondence
            notes the model was given, or the paragraphs it read.

Usage: python -m eval.answer_checks [--out results/answer_checks.md]
"""
import argparse
import json
import re
from collections import Counter

import pandas as pd

LOG = "data/rag/answers.jsonl"
# curly and straight quotes are paired separately: “ never closes with "
# pair every quote first, then keep the long ones: a length limit inside the regex would pair the closing
# quote of a short phrase ("opinion 1") with the opening quote of the next one
CURLY = re.compile(r"“([^“”]*)”")
STRAIGHT = re.compile(r'"([^"]*)"')
SECTION = re.compile(r"\b(?:[Ss]ections?|[Ss]s?\.|[Aa]rticles?|[Aa]rt\.)\s*(\d{1,3}[A-Z]{0,2})")


def norm(t: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", t.lower())) + " "


def pid_of(label: str) -> str:
    m = re.match(r"^(.*) ¶(\d+)(?: \(opinion (\d+)\))?$", label)
    return f"{m[1]}/{m[3] or 0}/{m[2]}"


def quote_pieces(q: str) -> list[str]:
    return [p for p in re.split(r"\s*(?:…|\.\.\.)\s*", q) if len(p.split()) >= 4]


def check(rec: dict, text_of: dict) -> dict:
    paras = [text_of.get(pid_of(l), "") for l in rec["retrieved"]]
    normed = [norm(p) for p in paras]
    every = " ".join(normed)
    out = Counter()
    misses = []
    for b in rec["blocks"]:
        cited = [normed[i] for i in set(b["cites"]) if i < len(normed)]
        # a cited passage that is itself verbatim text is a quotation (the web UI marks it with quotation marks)
        body = norm(b["text"])
        if cited and len(body.split()) >= 5:
            if any(body in c for c in cited):
                out["verbatim_cited"] += 1
            elif body in every:
                out["verbatim_other"] += 1
                misses.append(("verbatim passage matches a paragraph it does not cite", b["text"].strip()))
            else:
                out["paraphrase"] += 1
        t = b["text"]
        # a quote can open in one block and close in the next: drop the tail of a quote this block only closes
        first_close, first_open = t.find("”"), t.find("“")
        if first_close >= 0 and (first_open < 0 or first_close < first_open):
            t = t[first_close + 1:]
        quotes = [q for q in CURLY.findall(t) + STRAIGHT.findall(CURLY.sub(" ", t)) if len(q) >= 12]
        for q in quotes:
            for piece in quote_pieces(q):
                n = norm(piece)
                if cited and any(n in c for c in cited):
                    out["exact"] += 1
                elif n in every:
                    out["other_source" if cited else "exact_uncited"] += 1
                    if cited:
                        misses.append(("other_source", piece))
                else:
                    out["not_found"] += 1
                    misses.append(("not_found", piece))
    prose = [b for b in rec["blocks"] if b["text"].strip() and not re.match(r"^\s*#{1,6}\s", b["text"])]
    total = sum(len(b["text"].strip()) for b in prose) or 1
    covered = sum(len(b["text"].strip()) for b in prose if b["cites"])
    answer = "".join(b["text"] for b in rec["blocks"])
    allowed = set(SECTION.findall(rec["question"]))
    for line in rec.get("provision_map", []) + rec.get("related_map", []):
        allowed |= set(re.findall(r"s\.(\d{1,3}[A-Z]{0,2})", line))
    for p in paras:
        allowed |= set(SECTION.findall(p)) | set(re.findall(r"\b(\d{1,3}[A-Z]{0,2})\b", p))
    stated = set(SECTION.findall(answer)) | set(re.findall(r"\bs\.\s?(\d{1,3}[A-Z]{0,2})", answer))
    unsupported = sorted(stated - allowed)
    return {"question": rec["question"], "model": rec["model"], "quotes": dict(out), "misses": misses,
            "coverage": covered / total, "sections_stated": len(stated), "unsupported_sections": unsupported}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/answer_checks.md")
    args = ap.parse_args()
    recs = [r for r in map(json.loads, open(LOG)) if r.get("blocks")]
    paras = pd.read_parquet("data/graph/nodes_paragraph.parquet")
    text_of = dict(zip(paras["pid"], paras["text"]))
    rows = [check(r, text_of) for r in recs]

    tot = Counter()
    for r in rows:
        tot.update(r["quotes"])
    quoted = tot["exact"] + tot["other_source"] + tot["not_found"]
    lines = ["# Free checks on logged answers", "",
             f"`python -m eval.answer_checks`: {len(rows)} answers that kept their raw blocks "
             f"({sum(r['model'] == 'claude-opus-5-5' for r in rows)} Opus 5.5, "
             f"{sum(r['model'] == 'claude-haiku-5-5' for r in rows)} Haiku 5.5). Deterministic text checks, no model calls.", "",
             "| answer | model | verbatim cited passages | paraphrased cited passages | quotes exact / checked | quotes not found | cited prose | sections stated | unsupported sections |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        q = r["quotes"]
        n = q.get("exact", 0) + q.get("other_source", 0) + q.get("not_found", 0)
        wrong = f" (+{q['verbatim_other']} wrong source)" if q.get("verbatim_other") else ""
        lines.append(f"| {r['question'][:60]}{'…' if len(r['question']) > 60 else ''} | {r['model'].split('-')[1]} | "
                     f"{q.get('verbatim_cited', 0)}{wrong} | "
                     f"{q.get('paraphrase', 0)} | {q.get('exact', 0)} / {n} | {q.get('not_found', 0) + q.get('other_source', 0)} | "
                     f"{r['coverage']:.0%} | {r['sections_stated']} | {', '.join(r['unsupported_sections']) or '-'} |")
    passages = tot["verbatim_cited"] + tot["verbatim_other"] + tot["paraphrase"]
    lines += ["", f"**Cited passages:** {passages}; {tot['verbatim_cited']} are word-for-word text of the paragraph they cite, "
              f"{tot['verbatim_other']} are word-for-word text of a different retrieved paragraph (misattributed), "
              f"{tot['paraphrase']} are paraphrase or commentary (not checkable without a model)."]
    lines += ["", f"**Explicit quotation marks:** {quoted} quoted spans in cited passages: {tot['exact']} exact in the cited paragraph "
              f"({tot['exact'] / max(quoted, 1):.1%}), {tot['other_source']} found only in another retrieved paragraph, "
              f"{tot['not_found']} not found word for word. {tot['exact_uncited']} quotes in uncited passages, all found "
              f"in the retrieved paragraphs." if tot["exact_uncited"] else ""]
    misses = [(r["question"][:40], kind, piece) for r in rows for kind, piece in r["misses"]]
    if misses:
        lines += ["", "Quotes not matched in their cited paragraph:", ""]
        lines += [f"- *{q}…* ({kind}): “{p[:140]}{'…' if len(p) > 140 else ''}”" for q, kind, p in misses]
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
