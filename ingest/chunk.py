"""Cached lines -> clean -> paragraph chunks.

Paragraphs follow the judgment's own numbering ("12. ..."): the best strictly increasing chain of line-initial
numbers, so quoted lists / statute sub-clauses / deposition items stay inside their paragraph. Text before
para 1 (author line, case caption) is para_no 0. Judgments with no usable numbering fall back to layout
paragraphs (vertical gaps), numbered sequentially with numbered=False. Footnotes (2023+ citations live
there) attach to the paragraph whose superscript marks them.

Writes:
  data/processed/paragraphs.parquet  case_id, opinion, para_no, text, footnotes, page, numbered, path
  data/processed/cases.parquet       case_id, path, author, headnote, judgment, annexure, split_rule,
                                     n_opinions, n_paras, numbered
Separate opinions restart numbering, so (case_id, opinion, para_no) is the paragraph key.

Usage: python -m ingest.chunk [--workers 8]
"""
import argparse
import re
from multiprocessing import Pool

import pandas as pd

from ingest.clean import footnotes, join_lines, page_lines, split_case
from ingest.pdf_to_text import load

OUT = "data/processed"
# optional OCR junk ("· 2.") but not an opening quote (“23. = quoted paragraph of another judgment),
# then a number where OCR may read 1/0 as I/l/O ("I0." = 10)
# "5.30 p.m." / "10.12.1997" are not paragraph numbers: no digit right after the dot
PARA = re.compile(r"^[^\w(\"'“‘”’]{0,3}\s*([\dIlO]{1,3})\s*[.)](?!\d)\s*(?=\S)")
OCR_DIGITS = str.maketrans("IlO", "110")
GAP = 1.0        # vertical gap (in body-line heights) that starts a layout paragraph
PARA_GAP = 0.12  # paragraph starts sit ~0.2-0.5 below the previous line; continuations / list items ~-0.1-0.05
SKIP_COST = 0.6  # per paragraph number missing between two chosen starts
WEAK = 0.5       # value of a candidate without a clear vertical break (vs 1.0)


def _candidates(lines: list[dict]) -> list[tuple[int, int, float, int]]:
    out = []
    for i, l in enumerate(lines):
        m = PARA.match(l["text"])
        if not m:
            continue
        tok = m.group(1)
        if not any(c.isdigit() for c in tok) and tok not in "Il":
            continue
        n = int(tok.translate(OCR_DIGITS))
        if n == 0:
            continue
        out.append((i, n, 1.0 if l["first"] or l["gap"] >= PARA_GAP else WEAK, m.end()))
    return out


def _best_chain(cands: list[tuple[int, int, float, int]]) -> list[int]:
    """Max-score subsequence with strictly increasing numbers (quoted lists / dates / deposition items
    can't hijack the numbering: a false jump forfeits all the real paragraphs it skips)."""
    best, back = [], []
    for k, (_, n, w, _) in enumerate(cands):
        s, b = w - SKIP_COST * (n - 1), -1
        for j in range(k):
            nj = cands[j][1]
            if nj < n:
                v = best[j] + w - SKIP_COST * (n - nj - 1)
                if v > s:
                    s, b = v, j
        best.append(s)
        back.append(b)
    if not best:
        return []
    k = max(range(len(best)), key=best.__getitem__)
    chain = []
    while k != -1:
        chain.append(k)
        k = back[k]
    return chain[::-1]


def numbered_paras(lines: list[dict]) -> list[dict]:
    cands = _candidates(lines)
    starts = {cands[k][0]: cands[k] for k in _best_chain(cands)}
    paras, cur = [], {"para_no": 0, "page": lines[0]["page"] if lines else 0, "lines": [], "fns": []}
    for i, l in enumerate(lines):
        if i in starts:
            _, n, _, end = starts[i]
            if n == 2 and not paras and cur["lines"]:  # para 1 number absent from text layer
                cur["para_no"] = 1
            paras.append(cur)
            cur = {"para_no": n, "page": l["page"], "lines": [l["text"][end:]], "fns": []}
        else:
            cur["lines"].append(l["text"])
        cur["fns"] += [(l["page"], f) for f in l.get("fns", [])]
    paras.append(cur)
    return paras


OPINION = re.compile(r"^[A-Z][A-Za-z.' -]{2,60},\s*(?:C\.?\s?J\.?\s?I|J)\.?\s*(?:\([^)]{0,40}\))?\s*:?\s*$")
ANNEX = re.compile(r"^\s*(?:ANNEXURE|APPENDIX|SCHEDULE)\b[\s\-–:]*[A-Z0-9IVX\-]*\s*$", re.I)


def split_annex(lines: list[dict]) -> tuple[list[dict], str]:
    for i, l in enumerate(lines):
        if i > 0 and ANNEX.match(l["text"]):
            return lines[:i], join_lines([x["text"] for x in lines[i:]])
    return lines, ""


def split_opinions(lines: list[dict]) -> list[list[dict]]:
    """Separate opinions restart numbering at 1 under an 'X, J.' header line."""
    cuts = [0]
    for i, l in enumerate(lines):
        if i > 0 and OPINION.match(l["text"]):
            nxt = [PARA.match(x["text"]) for x in lines[i + 1:i + 5]]
            if any(m and m.group(1) in ("1", "I", "l") for m in nxt) and i - cuts[-1] > 10:
                cuts.append(i)
    return [lines[a:b] for a, b in zip(cuts, cuts[1:] + [len(lines)])]


def layout_paras(lines: list[dict]) -> list[dict]:
    paras = []
    for l in lines:
        if not paras or l["gap"] > GAP:
            paras.append({"para_no": len(paras) + 1, "page": l["page"], "lines": [], "fns": []})
        paras[-1]["lines"].append(l["text"])
        paras[-1]["fns"] += [(l["page"], f) for f in l.get("fns", [])]
    return paras


def process(row: tuple[str, str]) -> tuple[dict, list[dict]]:
    case_id, path = row
    pages = load(path)
    notes = footnotes(pages)
    r = split_case(page_lines(pages))
    jl, annex = split_annex(r["judgment_lines"])
    opinions = split_opinions(jl) if jl else []
    rows, numbered = [], bool(opinions)
    for op, seg in enumerate(opinions):
        paras = numbered_paras(seg)
        seg_numbered = len(paras) >= 3  # para 0 + at least 2 numbered paragraphs
        if not seg_numbered:
            paras, numbered = layout_paras(seg), False
        rows += [{"case_id": case_id, "opinion": op, "para_no": p["para_no"], "text": join_lines(p["lines"]),
                  "footnotes": [notes[k] for k in p["fns"] if k in notes], "page": p["page"],
                  "numbered": seg_numbered, "path": path} for p in paras]
    rows = [x for x in rows if x["text"]]
    case = {"case_id": case_id, "path": path, "author": r["author"], "headnote": r["headnote"],
            "judgment": "\n\n".join(x["text"] for x in rows), "annexure": annex, "split_rule": r["rule"],
            "n_opinions": len(opinions), "n_paras": len(rows), "numbered": numbered}
    return case, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    meta = pd.read_parquet("data/raw/meta.parquet")
    meta = meta[meta["ok"]]
    # case_id (neutral citation) is the graph key; fall back to SCR path when missing / duplicated
    cid = meta["case_id"].where(meta["case_id"].notna() & ~meta["case_id"].duplicated(keep=False), meta["path"])
    with Pool(args.workers) as pool:
        results = pool.map(process, list(zip(cid, meta["path"])), chunksize=8)
    cases = pd.DataFrame([c for c, _ in results])
    paras = pd.DataFrame([p for _, ps in results for p in ps])
    cases.to_parquet(f"{OUT}/cases.parquet", index=False)
    paras.to_parquet(f"{OUT}/paragraphs.parquet", index=False)
    print(f"cases={len(cases)} paragraphs={len(paras)}")
    print("split_rule:", cases["split_rule"].value_counts().to_dict())
    print(f"numbered={cases['numbered'].mean():.3f} author_found={cases['author'].notna().mean():.3f}")
    print("para chars:", paras["text"].str.len().describe(percentiles=[.5, .95, .99]).round().to_dict())


if __name__ == "__main__":
    main()
