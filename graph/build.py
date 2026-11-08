"""Processed tables -> graph node / edge tables (parquet), independent of Neo4j.

Nodes (data/graph/nodes_*.parquet):
  Case       case_id, title, petitioner, respondent, decision_date, year, citation, disposal, in_corpus, keys
             in-corpus judgments + external cited cases (one node per citation cluster)
  Paragraph  pid, case_id, opinion, para_no, text, footnotes
  Provision  pid, act, section
  Act        name
  Judge      key, name
  Court      name
Edges (data/graph/edges_*.parquet):
  CITES       src case_id -> dst case_id   {paras, n, in_footnote}
  INVOKES     case_id -> provision pid      {paras, n, subsections}
  PART_OF     provision pid -> act name
  MAPS_TO     old provision -> new: IPC->BNS, CrPC->BNSS, IEA->BSA  {new_sub, confidence}
  AUTHORED    judge key -> case_id
  SAT_ON      judge key -> case_id
  DECIDED_BY  case_id -> court
  HAS_PARAGRAPH case_id -> pid

External cases: parallel citations ("(2017) 8 SCC 570 : [2017] 8 SCR 151") are unioned across all judgments
and all headnote case-law lists, so an SCC cite resolves to an in-corpus case when any parallel key does.

Usage: python -m graph.build
"""
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from eval.citation_recall import headnote_groups
from link.citations import corpus_index, resolve

P, OUT = "data/processed", "data/graph"
COURT = "Supreme Court of India"
IPC, BNS = "Indian Penal Code, 1860", "Bharatiya Nyaya Sanhita, 2023"
CRPC, BNSS = "Code of Criminal Procedure, 1973", "Bharatiya Nagarik Suraksha Sanhita, 2023"
IEA, BSA = "Indian Evidence Act, 1872", "Bharatiya Sakshya Adhiniyam, 2023"
RELIABLE_ACT = {"explicit", "alias", "default", "default_case", "default_ipc", "article", "case"}  # paragraph-inferred Acts are ~56% precise: dropped


class DSU:
    """Union-find over citation keys that refuses merges putting two different keys of one reporter in a
    cluster (a case has one SCC cite) -- guards against ':' chaining distinct cases ("(1980) 1 SCC 81 : (1980) 1 SCC 98")."""

    def __init__(self):
        self.p, self.reps = {}, {}

    def find(self, x):
        if x not in self.p:
            self.p[x] = x
            self.reps[x] = {x.split("|")[0]: x}
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        A, B = self.reps[ra], self.reps[rb]
        if any(r in B and not compatible(B[r], k) for r, k in A.items()):
            return
        self.p[ra] = rb
        for r, k in A.items():  # keep the more specific volume ("Supp1" over "Supp")
            if r not in B or len(k) > len(B[r]):
                B[r] = k
        del self.reps[ra]


def compatible(a: str, b: str) -> bool:
    """Same cite, or bare 'Supp' vs 'SuppN' at the same year + page ("(1992) Supp SCC 335" = "(1992) Supp (1) SCC 335")."""
    if a == b:
        return True
    ra, ya, va, pa = a.split("|")
    rb, yb, vb, pb = b.split("|")
    return (ra, ya, pa) == (rb, yb, pb) and {va[:4], vb[:4]} == {"Supp"} and "Supp" in (va, vb)


def judge_key(name: str) -> str:
    return re.sub(r"[^A-Z]", "", name.upper())


def bench(headnote: str) -> list[str]:
    m = re.search(r"\[([A-Z][A-Z .,'-]{3,200}?),?\s*(?:C\.?J\.?I\.?\s*,?\s*)?J\s*J?\s*\.?\s*\]", headnote or "")
    if not m:
        return []
    names = re.split(r"\s*,\s*|\s+AND\s+", m[1])
    return [n.strip(" .") for n in names if len(judge_key(n)) >= 4 and n.strip(" .") not in ("CJI", "JJ", "J")]


def clean_name(name: str) -> str | None:
    """Headnote text before a citation -> "Party v. Party" (OCR-tolerant), else None."""
    name = re.sub(r"^(?:and|In|See|also|Case Law (?:Cited|Reference))\s+", "", name.strip(" ,.:;-–[("))
    if not re.search(r"\s(?:v\.?|vs\.?|versus)\s", name, re.I) or not 6 <= len(name) <= 150:
        return None
    return re.sub(r"\s+(?:vs\.?|versus|v)\s+", " v. ", name, flags=re.I)


def cluster_names(cases: pd.DataFrame, key_to: dict) -> dict:
    votes = defaultdict(Counter)
    for h in cases["headnote"]:
        for g in headnote_groups(h):
            n = clean_name(g["name"])
            cids = {key_to[k] for k in g["keys"] if k in key_to}
            if n and len(cids) == 1:
                votes[cids.pop()][n] += 1
    return {c: v.most_common(1)[0][0] for c, v in votes.items()}


def cluster_citations(cites: pd.DataFrame, cases: pd.DataFrame, exact, ranges) -> tuple[dict, dict]:
    """-> key -> cluster id (in-corpus case_id or 'ext:<min key>'), cluster id -> sorted keys."""
    dsu = DSU()
    for _, g in cites.groupby("group"):
        ks = g["key"].tolist()
        for k in ks:
            dsu.find(k)
        for a, b in zip(ks, ks[1:]):
            dsu.union(a, b)
    for h in cases["headnote"]:
        for g in headnote_groups(h):
            ks = sorted(g["keys"])
            for a, b in zip(ks, ks[1:]):
                dsu.union(a, b)
    members = defaultdict(set)
    for k in list(dsu.p):
        members[dsu.find(k)].add(k)
    key_to, cluster_keys = {}, {}
    for root, ks in members.items():
        hits = {resolve({"key": k, "reporter": k.split("|")[0], "year": int(k.split("|")[1]),
                         "volume": k.split("|")[2], "page": k.split("|")[3]}, exact, ranges) for k in ks} - {None}
        cid = sorted(hits)[0] if hits else "ext:" + min(ks)
        cluster_keys.setdefault(cid, set()).update(ks)
        for k in ks:
            key_to[k] = cid
    return key_to, {c: sorted(v) for c, v in cluster_keys.items()}


def case_tables():
    """Shared by main() and link.resolve: citation clusters + Case nodes (before name-based linking)."""
    cases = pd.read_parquet(f"{P}/cases.parquet")
    paras = pd.read_parquet(f"{P}/paragraphs.parquet")
    cites = pd.read_parquet(f"{P}/citations.parquet")
    meta = pd.read_parquet("data/raw/meta.parquet")
    exact, ranges = corpus_index(meta, cases)
    meta = meta.merge(cases[["path", "case_id", "headnote", "author"]], on="path", suffixes=("_meta", ""))
    key_to, cluster_keys = cluster_citations(cites, cases, exact, ranges)
    corpus = pd.DataFrame({
        "case_id": meta["case_id"], "title": meta["title"], "petitioner": meta["petitioner"],
        "respondent": meta["respondent"], "decision_date": pd.to_datetime(meta["decision_date"], format="%d-%m-%Y",
                                                                          errors="coerce").dt.strftime("%Y-%m-%d"),
        "year": meta["year"].astype(int), "citation": meta["citation"], "disposal": meta["disposal_nature"],
        "in_corpus": True})
    corpus["keys"] = corpus["case_id"].map(lambda c: cluster_keys.get(c, []))
    ext_ids = sorted(c for c in cluster_keys if c.startswith("ext:"))
    names = cluster_names(cases, key_to)
    external = pd.DataFrame({"case_id": ext_ids, "in_corpus": False, "keys": [cluster_keys[c] for c in ext_ids],
                             "year": [int(c.split("|")[1]) for c in ext_ids],
                             "title": [names.get(c) for c in ext_ids]})
    case_nodes = pd.concat([corpus, external], ignore_index=True)
    titles = {**dict(zip(corpus["case_id"], corpus["petitioner"].astype(str) + " v. " + corpus["respondent"].astype(str))),
              **names}
    return meta, paras, cites, key_to, case_nodes, titles


def load_for_resolve():
    _, paras, cites, key_to, case_nodes, titles = case_tables()
    return paras, cites, key_to, case_nodes, titles


def apply_links(case_nodes, cite_edges, res) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Name-based links from link.resolve: external->corpus and external duplicate merges, '(supra)' paragraphs,
    and names for external cases that had none."""
    remap = dict(res["dedupe"])
    c = res["corpus"]
    if len(c):
        remap.update(dict(zip(c.loc[c["target"].notna(), "case_id"], c.loc[c["target"].notna(), "target"])))
    for k in list(remap):  # dedupe target may itself map to a corpus case
        while remap[k] in remap and remap[remap[k]] != remap[k]:
            remap[k] = remap[remap[k]]

    nodes = case_nodes.copy()
    gone = nodes["case_id"].isin(remap)
    for src, dst in remap.items():  # carry keys over to the surviving node
        keys = nodes.loc[nodes["case_id"] == src, "keys"]
        if len(keys):
            i = nodes.index[nodes["case_id"] == dst]
            nodes.at[i[0], "keys"] = sorted(set(nodes.at[i[0], "keys"]) | set(keys.iloc[0]))
    nodes = nodes[~gone].reset_index(drop=True)
    names = res["names"]
    missing = nodes["title"].isna()
    nodes.loc[missing, "title"] = nodes.loc[missing, "case_id"].map(
        lambda c: max(names.get(c, [None]), key=lambda n: (n is not None, len(n or ""))))

    e = cite_edges.assign(dst=cite_edges["dst"].map(lambda d: remap.get(d, d)))
    s = res["supra"]
    s = s[s["target"].notna()] if len(s) else s
    sup = s.assign(dst=s["target"].map(lambda d: remap.get(d, d))).groupby(["case_id", "dst"])["para"] \
        .agg(lambda x: sorted(set(x))).rename("supra_paras").reset_index().rename(columns={"case_id": "src"}) if len(s) else None
    e = e[e["src"] != e["dst"]]
    e = e.groupby(["src", "dst"]).agg(paras=("paras", lambda x: sorted(set().union(*x))), n=("n", "sum"),
                                       in_footnote=("in_footnote", "any")).reset_index()
    if sup is not None:
        e = e.merge(sup, on=["src", "dst"], how="left")
        e["supra_paras"] = e["supra_paras"].map(lambda x: x if isinstance(x, list) else [])
        e["paras"] = [sorted(set(a) | set(b)) for a, b in zip(e["paras"], e["supra_paras"])]
    return nodes, e


def main():
    from link.resolve import resolve_all
    os.makedirs(OUT, exist_ok=True)
    provs = pd.read_parquet(f"{P}/provisions.parquet")
    meta, paras, cites, key_to, case_nodes, titles = case_tables()
    corpus = case_nodes[case_nodes["in_corpus"]]

    # --- CITES (aggregated per case pair; paras as "opinion:para_no")
    cites = cites.assign(dst=cites["key"].map(key_to), para=cites["opinion"].astype(str) + ":" + cites["para_no"].astype(str))
    grp = cites.drop_duplicates(["case_id", "group"]).query("dst != case_id")
    cite_edges = grp.groupby(["case_id", "dst"]).agg(
        paras=("para", lambda s: sorted(set(s))), n=("group", "size"), in_footnote=("in_footnote", "any")).reset_index()
    cite_edges = cite_edges.rename(columns={"case_id": "src"})

    # --- name-based linking (fuzzy + cached Claude Haiku decisions; python -m link.resolve --llm fills the cache)
    res = resolve_all(paras, cites, key_to, case_nodes, titles, use_llm=False)
    case_nodes, cite_edges = apply_links(case_nodes, cite_edges, res)
    print("supra:", res["supra"]["method"].value_counts().to_dict(),
          "| external->corpus:", res["corpus"]["method"].value_counts().to_dict(), "| merged dups:", len(res["dedupe"]))

    # --- provisions
    pv = provs[provs["act_source"].isin(RELIABLE_ACT) & provs["act"].notna()].copy()
    pv["pid"] = pv["act"] + "|" + pv["section"]
    pv["para"] = pv["opinion"].astype(str) + ":" + pv["para_no"].astype(str)
    invokes = pv.groupby(["case_id", "pid"]).agg(
        paras=("para", lambda s: sorted(set(s))), n=("pid", "size"),
        subsections=("subsection", lambda s: sorted(set(x for x in s if x)))).reset_index()
    prov_nodes = pv.drop_duplicates("pid")[["pid", "act", "section"]]

    # old Code -> new Sanhita: IPC -> BNS (link/ipc_bns.py), CrPC -> BNSS and Evidence Act -> BSA (link/law_map.py).
    # New-Act targets become Provision nodes even if no judgment cites them yet; only old sections the corpus
    # actually invokes are mapped.
    frames = []
    m = pd.read_csv("data/ref/ipc_bns.csv", dtype=str).fillna("")
    frames.append(pd.DataFrame({"src": IPC + "|" + m["ipc"], "dst": BNS + "|" + m["bns"],
                                "new_sub": m["bns_sub"], "confidence": m["confidence"]}))
    for csv, old_act, new_act in [("crpc_bnss", CRPC, BNSS), ("iea_bsa", IEA, BSA)]:
        m = pd.read_csv(f"data/ref/{csv}.csv", dtype=str).fillna("")
        frames.append(pd.DataFrame({"src": old_act + "|" + m["old"], "dst": new_act + "|" + m["new"],
                                    "new_sub": m["new_sub"], "confidence": m["confidence"]}))
    maps_to = pd.concat(frames, ignore_index=True)
    maps_to = maps_to[maps_to["src"].isin(set(prov_nodes["pid"]))]
    new_nodes = pd.DataFrame({"pid": maps_to["dst"].unique()})
    new_nodes = new_nodes[~new_nodes["pid"].isin(set(prov_nodes["pid"]))].assign(
        act=lambda d: d["pid"].str.split("|").str[0], section=lambda d: d["pid"].str.split("|").str[1])
    prov_nodes = pd.concat([prov_nodes, new_nodes], ignore_index=True)

    act_nodes = pd.DataFrame({"name": sorted(prov_nodes["act"].unique())})
    part_of = prov_nodes[["pid", "act"]]

    # --- judges: metadata 'judge' = authoring judge(s); bench from headnote "[A AND B, JJ.]"
    authored, sat_on, names = [], [], {}
    for r in meta.itertuples():
        for n in str(r.judge or "").split(","):
            if n.strip():
                names.setdefault(judge_key(n), n.strip().title())
                authored.append((judge_key(n), r.case_id))
        for n in bench(r.headnote):
            names.setdefault(judge_key(n), n.title())
            sat_on.append((judge_key(n), r.case_id))
    judge_nodes = pd.DataFrame(sorted(names.items()), columns=["key", "name"])
    authored = pd.DataFrame(authored, columns=["judge", "case_id"]).drop_duplicates()
    sat_on = pd.DataFrame(sat_on, columns=["judge", "case_id"]).drop_duplicates()

    # --- paragraphs
    para_nodes = paras.assign(pid=paras["case_id"] + "/" + paras["opinion"].astype(str) + "/" + paras["para_no"].astype(str))[
        ["pid", "case_id", "opinion", "para_no", "text", "footnotes"]]

    tables = {
        "nodes_case": case_nodes, "nodes_paragraph": para_nodes, "nodes_provision": prov_nodes,
        "nodes_act": act_nodes, "nodes_judge": judge_nodes, "nodes_court": pd.DataFrame({"name": [COURT]}),
        "edges_cites": cite_edges, "edges_invokes": invokes, "edges_part_of": part_of, "edges_maps_to": maps_to,
        "edges_authored": authored, "edges_sat_on": sat_on,
        "edges_decided_by": pd.DataFrame({"case_id": corpus["case_id"], "court": COURT}),
        "edges_has_paragraph": para_nodes[["case_id", "pid"]],
    }
    for name, df in tables.items():
        df.to_parquet(f"{OUT}/{name}.parquet", index=False)
        print(f"{name:22s} {len(df):>8,}")
    internal = cite_edges["dst"].isin(set(corpus["case_id"]))
    print(f"CITES to in-corpus cases: {internal.sum():,} ({internal.mean():.3f}); "
          f"in-corpus cases cited at least once: {cite_edges.loc[internal, 'dst'].nunique():,}")
    print(f"cases with bench parsed: {sat_on['case_id'].nunique() / len(corpus):.3f}")


if __name__ == "__main__":
    main()
