"""Silver recall of link/citations.py against the SCR editors' headnote case-law lists.

Each headnote citation group (parallel cites "(2003) 9 SCC 468 : [2003] 3 SCR 12") is a case the judgment cites.
  found_by_cite  any parallel key appears among the judgment's extracted citations
  name_only      not found by cite, but the party name preceding it appears in the judgment text
                 (cited as "Romila Thapar (supra)" / name only -> needs fuzzy/LLM linking)
  missing        neither
Editors add parallel cites the judgment never prints, so found_by_cite is a lower bound.

Usage: python -m eval.citation_recall
"""
import re

import pandas as pd

from link.citations import extract


def headnote_groups(headnote: str) -> list[dict]:
    hits = extract(headnote)
    groups = {}
    for h in hits:
        groups.setdefault(h["group"], []).append(h)
    out, prev_end = [], 0
    for g in sorted(groups.values(), key=lambda x: x[0]["start"]):
        name = headnote[prev_end:g[0]["start"]]
        name = re.split(r"[;\n]|relied on|referred to|followed|distinguished|overruled", name)[-1]
        out.append({"keys": {h["key"] for h in g}, "name": re.sub(r"\s+", " ", name).strip(" ,.:-–")})
        prev_end = g[-1]["end"]
    return out


GENERIC_PARTY = {"state", "and", "ors", "anr", "the", "union", "india", "another", "others", "of"}


def name_in(name: str, text: str) -> bool:
    """First distinctive party ("Union of India v. Prafulla Kumar Samal" -> Prafulla Kumar) found in text."""
    for party in re.split(r"\s+(?:v\.|vs\.?|versus)\s+", name, flags=re.I):
        toks = [t for t in re.findall(r"[A-Za-z]{3,}", party) if t.lower() not in GENERIC_PARTY]
        if toks:
            return all(re.search(rf"\b{re.escape(t)}\b", text, re.I) for t in toks[:2])
    return False


def main():
    cases = pd.read_parquet("data/processed/cases.parquet")
    cites = pd.read_parquet("data/processed/citations.parquet")
    found_keys = cites.groupby("case_id")["key"].agg(set).to_dict()
    paras = pd.read_parquet("data/processed/paragraphs.parquet")
    full_text = (paras["text"] + " " + paras["footnotes"].map(" ".join)).groupby(paras["case_id"]).agg(" ".join).to_dict()
    rows = []
    for c in cases.itertuples():
        if c.split_rule == "none":  # headnote not separated from judgment
            continue
        for g in headnote_groups(c.headnote):
            if c.case_id in g["keys"] or len(g["name"]) < 4:
                continue
            got = g["keys"] & found_keys.get(c.case_id, set())
            status = "found_by_cite" if got else ("name_only" if name_in(g["name"], full_text[c.case_id]) else "missing")
            rows.append({"case_id": c.case_id, "era": "2023+" if c.path[:4] >= "2023" else "2016-22", "status": status})
    df = pd.DataFrame(rows)
    print(f"headnote-cited cases: {len(df)} across {df['case_id'].nunique()} judgments")
    print(pd.crosstab(df["era"], df["status"], normalize="index", margins=True).round(3).to_string())


if __name__ == "__main__":
    main()
