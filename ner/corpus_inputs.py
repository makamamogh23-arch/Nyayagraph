"""Build NER inputs from our own judgments for the fine-tuned Qwen adapter (run on Kaggle with ner/infer_vllm.py).

Two parts, each in the prompt format the adapter was trained on (ner/to_instruct.py):
  appearances  the headnote block from "APPELLATE JURISDICTION" to "delivered by <judge>": case numbers, the court
               appealed from, the lawyers on each side. Preamble-style text, so the preamble label set is used
               plus DATE and CASE_NUMBER (both trained on judgment text).
  witnesses    judgment paragraphs that name a numbered prosecution witness (PW-1, P.W.2 ...): witnesses, other
               persons, police stations, places. Judgment label set. Long paragraphs are split at sentence
               boundaries into pieces of at most MAX_CHARS, since training inputs were short.

Each row: id, part, case_id, opinion, para_no, offset (char offset of the piece in its source text), text, messages
(system + user, no answer). Offsets let ner/corpus_entities.py map predicted strings back to the source text.

Usage: python -m ner.corpus_inputs [--out data/ner_corpus] [--parts appearances witnesses]
"""
import argparse
import os
import re

import pandas as pd

from ner.common import JUDGEMENT_LABELS, PREAMBLE_LABELS, write_jsonl
from ner.to_instruct import SYSTEM, build_instruction

MAX_CHARS = 2000
PW = re.compile(r"\bP\.?\s?W\.?\s?-?\s?\d+")
TAIL_START = re.compile(r"(?:CRIMINAL|CIVIL)?\s*APPELLATE\s+JURISDICTION", re.I)
TAIL_END = re.compile(r"delivered\s+by\s*\n?[^\n]*", re.I)
APPEARANCE_LABELS = PREAMBLE_LABELS + ["DATE", "CASE_NUMBER"]
SENT_END = re.compile(r"(?<=[.;:?!])\s+(?=[\"“(]?[A-Z0-9])")


def pieces(text: str, limit: int = MAX_CHARS) -> list[tuple[int, str]]:
    """Split text into (offset, piece) at sentence ends, each piece at most `limit` chars where possible."""
    if len(text) <= limit:
        return [(0, text)]
    cuts = [m.end() for m in SENT_END.finditer(text)]
    out, start, last = [], 0, 0
    for c in cuts + [len(text)]:
        if c - start > limit and last > start:
            out.append((start, text[start:last]))
            start = last
        last = c
    out.append((start, text[start:]))
    # a single sentence longer than the limit: hard split on whitespace
    final = []
    for off, p in out:
        while len(p) > limit * 1.5:
            cut = p.rfind(" ", 0, limit) if p.rfind(" ", 0, limit) > 0 else limit
            final.append((off, p[:cut]))
            off, p = off + cut, p[cut:]
        final.append((off, p))
    return [(o, p) for o, p in final if p.strip()]


def row(id_: str, part: str, case_id: str, opinion: int, para_no: int, offset: int, text: str, labels) -> dict:
    instruction = build_instruction(labels)
    return {"id": id_, "part": part, "case_id": case_id, "opinion": int(opinion), "para_no": int(para_no),
            "offset": offset, "text": text,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"{instruction}\n\nText:\n{text}"}]}


def appearances(cases: pd.DataFrame) -> list[dict]:
    rows = []
    for case_id, head in zip(cases["case_id"], cases["headnote"].fillna("")):
        m = TAIL_START.search(head)
        if not m:
            continue
        end = TAIL_END.search(head, m.start())
        stop = end.end() if end else min(len(head), m.start() + 1500)
        for k, (off, p) in enumerate(pieces(head[m.start():stop])):
            rows.append(row(f"{case_id}|app|{k}", "appearances", case_id, 0, 0, m.start() + off, p, APPEARANCE_LABELS))
    return rows


def witnesses(paras: pd.DataFrame) -> list[dict]:
    rows = []
    hit = paras[paras["text"].str.contains(PW)]
    for r in hit.itertuples():
        for k, (off, p) in enumerate(pieces(r.text)):
            rows.append(row(f"{r.case_id}|{r.opinion}|{r.para_no}|{k}", "witnesses", r.case_id, r.opinion,
                            r.para_no, off, p, JUDGEMENT_LABELS))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ner_corpus")
    ap.add_argument("--parts", nargs="+", default=["appearances", "witnesses"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for part in args.parts:
        if part == "appearances":
            rows = appearances(pd.read_parquet("data/processed/cases.parquet", columns=["case_id", "headnote"]))
        else:
            rows = witnesses(pd.read_parquet("data/processed/paragraphs.parquet",
                                             columns=["case_id", "opinion", "para_no", "text"]))
        write_jsonl(f"{args.out}/{part}.jsonl", rows)
        chars = sum(len(r["text"]) for r in rows)
        print(f"{part}: {len(rows)} pieces from {len({r['case_id'] for r in rows})} cases, {chars:,} chars "
              f"-> {args.out}/{part}.jsonl")


if __name__ == "__main__":
    main()
