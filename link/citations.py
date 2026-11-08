"""Case-law citation extraction (SCC, SCC (Cri), SCC OnLine, SCR, AIR, INSC, Cri LJ) + exact in-corpus resolution.

Each hit is normalized to key = REPORTER|year|volume|page. Citations joined by ':' or '=' are parallel
citations of one case and share a group (group = case_id/opinion/para_no/index).

Writes data/processed/citations.parquet:
  case_id, opinion, para_no, in_footnote, group, reporter, year, volume, page, key, raw, start, end,
  target_case_id   (start/end are offsets into the paragraph text, or into the footnote when in_footnote)

Usage: python -m link.citations
"""
import re

import pandas as pd

R_SCC = r"(?:S\.?\s?C\.?\s?C\.?|sec|S\.C\.C)"
R_SCR = r"S\.?\s?C\.?\s?R\.?"
# "Supp 2 SCC" and "2 Suppl. SCR" both occur
VOL = r"(?P<supp>Supp(?:l)?\.?\s*(?:\(\s*\d\s*\)|\d)?\s*)?(?P<vol>\d{1,2})?\s*(?P<supp2>Supp(?:l)?\.?\s*)?"
SUB = r"(?:\s*\((?P<sub>Cri|Crl|L\s?&\s?S|Tax|Civ)\.?\))?"

PATTERNS = [
    # (2003) 9 SCC 468 · (2003) Supp 2 SCC 12 · [2016] 8 SCR 380 · (2004) 1 SCC (Cri) 123
    ("bracketed", re.compile(
        rf"[(\[]\s*(?P<year>(?:18|19|20)\d\d)\s*[)\]]\s*{VOL}\s*(?P<rep>{R_SCC}|{R_SCR}){SUB}\s*(?P<page>\d{{1,4}})\b")),
    # 2011 (8) SCR 123 · 2015 (12) SCC 45 · 2003 SCC (Cri) 1234
    ("year_first", re.compile(
        rf"\b(?P<year>(?:18|19|20)\d\d)\s*(?:\(\s*(?P<supp>Supp(?:l)?\.?\s*\d?\s*)?(?P<vol>\d{{1,2}})?\s*\))?\s*"
        rf"(?P<rep>{R_SCC}|{R_SCR}){SUB}\s*(?P<page>\d{{1,4}})\b")),
    ("scc_online", re.compile(
        r"\b(?P<year>(?:19|20)\d\d)\s*SCC\s*On\s*Line\s+(?P<court>[A-Z][A-Za-z&]{1,12})\s+(?P<page>\d{1,6})\b")),
    ("air", re.compile(  # no \b: OCR glues names onto it ("AnotherAIR 1965 SC 1887")
        r"(?<![A-Z])A\.?\s?I\.?\s?R\.?\s*\(?(?P<year>(?:19|20)\d\d)\)?\s*(?P<court>S\.?\s?C\.?|[A-Z][A-Za-z]{1,10})\s*(?P<page>\d{1,5})\b")),
    ("insc", re.compile(r"\b(?P<year>(?:19|20)\d\d)\s*INSC\s*(?P<page>\d{1,5})\b")),
    ("crilj", re.compile(r"\b(?P<year>(?:19|20)\d\d)\s*Cri\.?\s*L\.?\s*J\.?\s*(?P<page>\d{1,5})\b", re.I)),
]
PARALLEL_SEP = re.compile(r"^\s*[:=]\s*$")


def _reporter(kind: str, m: re.Match) -> tuple[str, str]:
    g = m.groupdict()
    if kind in ("bracketed", "year_first"):
        rep = "SCC" if re.fullmatch(R_SCC, g["rep"], re.I) else "SCR"
        if g.get("sub"):
            rep += f" ({re.sub(r'Crl', 'Cri', g['sub'].replace(' ', ''))})"
        vol = g.get("vol") or ""
        if g.get("supp") or g.get("supp2"):  # "Supp (3)" / "Supp. 3" / "3 Suppl." -> Supp3
            vol = "Supp" + re.sub(r"\D", "", g.get("supp") or "") + vol
        return rep, vol
    if kind == "scc_online":
        return f"SCC OnLine {g['court']}", ""
    if kind == "air":
        court = re.sub(r"\W", "", g["court"]).upper()
        return f"AIR {'SC' if court in ('SC', 'SCC') else g['court']}", ""  # "AIR 1992 SCC 604" = typo for SC
    return {"insc": "INSC", "crilj": "Cri LJ"}[kind], ""


def extract(text: str) -> list[dict]:
    hits = []
    for kind, pat in PATTERNS:
        for m in pat.finditer(text):
            rep, vol = _reporter(kind, m)
            hits.append({"reporter": rep, "year": int(m["year"]), "volume": vol, "page": m["page"],
                         "raw": m.group(0), "start": m.start(), "end": m.end()})
    # overlapping matches from different patterns: keep the longest
    hits.sort(key=lambda h: (h["start"], -(h["end"] - h["start"])))
    out = []
    for h in hits:
        if out and h["start"] < out[-1]["end"]:
            continue
        out.append(h)
    g = 0
    for i, h in enumerate(out):
        if i and not PARALLEL_SEP.match(text[out[i - 1]["end"]:h["start"]]):
            g += 1
        h["group"] = g
        h["key"] = f"{h['reporter']}|{h['year']}|{h['volume']}|{h['page']}"
    return out


def corpus_index(meta: pd.DataFrame, cases: pd.DataFrame) -> tuple[dict, list]:
    """INSC + SCR keys of in-corpus cases. SCR also indexes the page range so pinpoint cites resolve."""
    m = meta.merge(cases[["path", "case_id"]], on="path", suffixes=("_meta", ""))
    exact, ranges = {}, []
    for r in m.itertuples():
        if isinstance(r.case_id_meta, str):
            for h in extract(r.case_id_meta):
                exact[h["key"]] = r.case_id
        y, vol, p0, p1 = (r.path.split("_") + [None] * 4)[:4]
        if p1 and p0.isdigit() and p1.isdigit():
            ranges.append((int(y), vol, int(p0), int(p1), r.case_id))
    return exact, ranges


def resolve(h: dict, exact: dict, ranges: list) -> str | None:
    if h["key"] in exact:
        return exact[h["key"]]
    if h["reporter"] == "SCR" and h["page"].isdigit():
        p = int(h["page"])
        for y, vol, p0, p1, cid in ranges:
            if y == h["year"] and vol == h["volume"] and p0 <= p <= p1:
                return cid
    return None


def main():
    paras = pd.read_parquet("data/processed/paragraphs.parquet")
    cases = pd.read_parquet("data/processed/cases.parquet")
    meta = pd.read_parquet("data/raw/meta.parquet")
    exact, ranges = corpus_index(meta, cases)

    rows = []
    for r in paras.itertuples():
        # body text, then each footnote (2023+ judgments cite cases in footnotes)
        for src, text in [("body", r.text)] + [(f"fn{i}", t) for i, t in enumerate(r.footnotes)]:
            for h in extract(text):
                h.update(case_id=r.case_id, opinion=r.opinion, para_no=r.para_no, in_footnote=src != "body",
                         group=f"{r.case_id}/{r.opinion}/{r.para_no}/{src}/{h['group']}")
                h["target_case_id"] = resolve(h, exact, ranges)
                rows.append(h)
    df = pd.DataFrame(rows)
    df.loc[df["target_case_id"] == df["case_id"], "target_case_id"] = None  # self-citation = own header echo
    df.to_parquet("data/processed/citations.parquet", index=False)

    groups = df.groupby("group").agg(case_id=("case_id", "first"), target=("target_case_id", lambda s: s.dropna().iloc[0] if s.notna().any() else None))
    print(f"citations={len(df)} groups(cited cases)={len(groups)}")
    print(f"paragraphs with >=1 citation: {df[['case_id','opinion','para_no']].drop_duplicates().shape[0] / len(paras):.3f}")
    print(f"cases with >=1 citation: {df['case_id'].nunique() / cases.shape[0]:.3f}")
    print("by reporter:", df["reporter"].value_counts().head(12).to_dict())
    print(f"groups resolved to an in-corpus case (exact INSC/SCR): {groups['target'].notna().mean():.3f} "
          f"({groups['target'].notna().sum()})")


if __name__ == "__main__":
    main()
