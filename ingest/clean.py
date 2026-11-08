"""Cached line geometry -> clean text lines, split into headnote / judgment.

Removes: running headers + page numbers (top band), SCR margin letters A-H, footnotes and superscript
footnote markers (small font), split ligatures ("oﬀ ences"), end-of-line hyphenation; detects author judge.
"""
import re
import statistics
import unicodedata
from collections import Counter

TOP_BAND = 0.12      # fraction of page height holding running headers
SMALL_FONT = 0.8     # spans below this x body size are footnotes / superscripts
MARGIN_CHARS = set("ABCDEFGHabcdefgh8•.,-")  # OCR renders margin letters B/C as 8/c, etc.

NAME = r"([A-Z][A-Za-z.' -]{2,60}?)"
JUDGMENT_START = [
    ("delivered_by", re.compile(
        # OCR-tolerant: "delivered .by", "Coutt", "AMITAVAROY,J."
        # name is any OCR junk up to ", J." ("ARUN l\USHRA, J.")
        r"(?:The\s+)?(?:Judgment|Order)s?\s+of\s+the\s+Co\w{2,3}\s+(?:was|were)\s+(?:delivered|passed|pronounced)[\s.]*by\s*:?\s*"
        r"([^\n,]{3,60}?),?\s*(?:C\.?\s?J\.?\s?I\.?|J\.?)(?=\s|$|\d)", re.I)),
    ("order_delivered", re.compile(  # unsigned orders: "The following Order of the Court was delivered: ORDER"
        r"The\s+following\s+(?:Judgment|Order)\s+of\s+the\s+Co\w{2,3}\s+was\s+(?:delivered|passed)\s*:?\s*"
        r"(?:O\s*R\s*D\s*E\s*R|J\s*U\s*D\s*G\s*M\s*E\s*N\s*T)?()", re.I)),
    ("dscr_heading", re.compile(
        r"Judgment\s*/\s*Order\s+of\s+the\s+Supreme\s+Court\s+(?:Judgment|Order)s?\s+"
        r"(?:" + NAME + r",\s*(?:C\.?\s?J\.?\s?I\.?|J\.?)(?=\s|$))?", re.I)),
]
TAIL = re.compile(r"\s*(?:Headnotes prepared by|Result of the case)\b.*$", re.S)
OLD_TAIL = re.compile(r"\n[A-Z][\w.]*(?: [A-Z][\w.]*){0,3}\s*\n(?:Appeals?|Petitions?|Matters?)[^\n]{0,60}\.\s*$")


def _norm_header(t: str) -> str:
    return re.sub(r"[\d\W]+", "", t).lower()


def _body_size(pages: list[dict]) -> float:
    sizes = [sz for p in pages for l in p["lines"] for t, sz in l["spans"] for _ in range(len(t))]
    return statistics.median(sizes) if sizes else 10.0


def footnotes(pages: list[dict]) -> dict[tuple[int, int], str]:
    """Small-font lines in the lower half of a page -> {(page, n): text}. Digital SCR (2023+) puts
    case citations here ("4  [2017] 8 SCR 151 : (2017) 8 SCC 570")."""
    body, out = _body_size(pages), {}
    for pno, p in enumerate(pages):
        rows = {}
        for l in p["lines"]:
            if l["y0"] > 0.5 * p["h"] and all(sz < SMALL_FONT * body for _, sz in l["spans"]):
                rows.setdefault(round(l["y0"]), []).append((l["x0"], "".join(t for t, _ in l["spans"])))
        cur = None
        for y in sorted(rows):
            text = fix_text(" ".join(t for _, t in sorted(rows[y])))
            m = re.match(r"^(\d{1,3})\s+(.*)", text)
            if m:
                cur = (pno, int(m[1]))
                out[cur] = m[2]
            elif cur:
                out[cur] += " " + text
    return out


def page_lines(pages: list[dict]) -> list[dict]:
    """-> [{text, page, gap, dx, first, fns}] in reading order; gap = vertical gap before line / body line
    height; fns = footnote numbers marked by superscripts on the line (text itself drops them)."""
    body = _body_size(pages)
    # header band: top 12% of the page, or the first ~2 text rows (Letter-size scans sit lower)
    bands = [max(TOP_BAND * p["h"], min((l["y0"] for l in p["lines"]), default=0) + 2.6 * body) for p in pages]
    top = Counter()
    for p, band in zip(pages, bands):
        for t in {_norm_header("".join(s[0] for s in l["spans"])) for l in p["lines"] if l["y0"] < band}:
            top[t] += 1

    out = []
    for pno, (p, band) in enumerate(zip(pages, bands)):
        rows = []
        for l in p["lines"]:
            text = "".join(t for t, sz in l["spans"] if sz >= SMALL_FONT * body)
            s = text.strip()
            if not s:
                continue
            fns = [int(t) for t, sz in l["spans"] if sz < SMALL_FONT * body and re.fullmatch(r"\d{1,3}", t.strip())]
            if l["y0"] < band:
                n = _norm_header(s)
                if not n or top[n] >= 2 or "scr" in n or "supremecourtreports" in n:
                    continue
            if len(s) <= 2 and set(s) <= MARGIN_CHARS:
                continue
            if re.fullmatch(r"\d{1,4}", s) and (l["y0"] > 0.9 * p["h"] or l["y0"] < band):
                continue
            rows.append([l["y0"], l["y1"], l["x0"], text, fns])
        # merge fragments on the same row ("1.\t" + "Heard.")
        rows.sort(key=lambda r: (r[0], r[2]))
        merged = []
        for r in rows:
            if merged and abs(r[0] - merged[-1][0]) < 2.5:
                merged[-1][3] = merged[-1][3].rstrip() + " " + r[3].lstrip()
                merged[-1][4] = merged[-1][4] + r[4]
            else:
                merged.append(r)
        # dx: indent relative to the page's left text edge (odd/even pages have different margins)
        left = min((r[2] for r in merged), default=0.0)
        prev_y1 = None
        for y0, y1, x0, text, fns in merged:
            gap = (y0 - prev_y1) / body if prev_y1 is not None else 0.0
            out.append({"text": fix_text(text), "page": pno, "gap": round(gap, 2), "dx": round(x0 - left, 1),
                        "first": prev_y1 is None, "fns": fns})
            prev_y1 = y1
    return out


def fix_text(t: str) -> str:
    t = re.sub(r"([ﬀﬁﬂﬃﬄ])\s+(?=[a-z])", r"\1", t)
    t = unicodedata.normalize("NFKC", t).replace("\t", " ")
    return re.sub(r"[  ]+", " ", t).strip()


def join_lines(lines: list[str]) -> str:
    out = ""
    for s in lines:
        if out.endswith("-") and re.search(r"[a-z]-$", out) and s[:1].islower():
            out = out[:-1] + s
        else:
            out = f"{out} {s}" if out else s
    return out


def split_case(lines: list[dict]) -> dict:
    """-> {headnote, judgment_lines, author, rule}. judgment_lines keep page/gap for chunking."""
    text = "\n".join(l["text"] for l in lines)
    offsets, pos = [], 0
    for l in lines:
        offsets.append(pos)
        pos += len(l["text"]) + 1

    start, author, rule = None, None, "none"
    for name, pat in JUDGMENT_START:
        m = pat.search(text)
        if m:
            start, author, rule = m.end(), (m.group(1) or "").strip() or None, name
            break
    if start is None:  # after the counsel block / jurisdiction line, at the first "1." paragraph
        j = text.find("APPELLATE JURISDICTION")
        m = re.search(r"(?m)^\s*1\s*\.\s", text[j:]) if j != -1 else None
        if m:
            start, rule = j + m.start(), "first_para"
    if start is None:
        start = 0

    # first judgment line = line containing `start`; keep any text after the match on that line
    i = max(k for k, o in enumerate(offsets) if o <= start)
    head = text[:start].strip()
    jl = [dict(l) for l in lines[i:]]
    jl[0]["text"] = text[start:offsets[i] + len(lines[i]["text"])].strip()
    jl = [l for l in jl if l["text"]]

    body = "\n".join(l["text"] for l in jl)
    for pat in (TAIL, OLD_TAIL):
        m = pat.search(body)
        if m:
            body = body[:m.start()]
            keep = len(body)
            acc, cut = 0, []
            for l in jl:
                if acc >= keep:
                    break
                cut.append(dict(l, text=l["text"][:max(keep - acc, 0)]))
                acc += len(l["text"]) + 1
            jl = [l for l in cut if l["text"].strip()]
            break
    if author:
        author = re.sub(r"\s+", " ", author).strip(" .,").title()
    return {"headnote": head, "judgment_lines": jl, "author": author, "rule": rule}
