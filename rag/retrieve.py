"""GraphRAG retrieval over the judgment corpus: hybrid text search + Neo4j graph expansion.

  1. Query analysis: provisions named in the question (link.provisions parser), expanded across IPC <-> BNS
     (and CrPC <-> BNSS, IEA <-> BSA) through MAPS_TO; titles of corpus cases named in the question.
  2. Text search over paragraph chunks: BM25 (bm25s); dense (bge-base, rag/embed.py) is optional (use_dense) and off
     by default, since it lowered every metric on the benchmark. A paragraph scores as its best chunk.
  3. Graph re-rank: boost paragraphs that INVOKE a named provision (paragraph-level, from INVOKES.paras), cases
     that invoke it anywhere, and a small PageRank prior; at most PER_CASE paragraphs per case.
  4. Precedent expansion: corpus cases CITED from the top paragraphs (CITES.paras holds the citing paragraph);
     each contributes its paragraph most similar to the question.
  5. Scope (rag/scope.py): a lower court, lawyer or witness named in the question restricts the search to the
     judgments the graph links to it; a witness's own paragraphs are boosted. No precedent expansion when scoped.

Usage: python -m rag.retrieve "question"
"""
import math
import re
import sys
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from graph.conn import driver
from rag import scope as scoping

RAG = "data/rag"
K_TEXT, RRF_K = 100, 60
W_DENSE = 1.0  # weight of the dense list in reciprocal rank fusion (BM25 = 1)
RERANKER, RERANK_DEPTH = "BAAI/bge-reranker-base", 30  # cross-encoder over the top candidates
PER_CASE, TOP_PARAS, TOP_PRECEDENTS, MAX_WORDS = 3, 12, 4, 350
# Graph boosts, set from eval/rag_bench.py (results/rag_bench.md): provision boosts apply only to the old-Act
# *counterparts* of new-Act provisions the question names (BNSS 528 -> paragraphs invoking CrPC 482); BM25 already
# matches a literally named section. A PageRank prior showed no gain and is off.
W_PARA_PROV, W_CASE_PROV, W_PAGERANK, W_NAMED_CASE = 0.020, 0.008, 0.0, 0.030
W_WITNESS = 0.050  # paragraphs naming the witness a question asks about (rag/scope.py)
OLD_ACTS = {"Indian Penal Code, 1860", "Code of Criminal Procedure, 1973", "Indian Evidence Act, 1872"}
IPC, BNS = "Indian Penal Code, 1860", "Bharatiya Nyaya Sanhita, 2023"

# IPC punishment sections -> the IPC sections that define the offence (reviewed by hand). A question about
# "Section 103 BNS" (= IPC 302, punishment for murder) needs the definition in IPC 300 (= BNS 101) to be answered.
DEFINED_BY = {
    "302": ("299", "300"), "304": ("299", "300"), "307": ("300",), "308": ("299",),
    "306": ("107",), "120B": ("120A",), "147": ("146",), "148": ("146",),
    "323": ("319",), "324": ("319",), "325": ("320",), "326": ("320",),
    "341": ("339",), "342": ("340",), "352": ("350",), "353": ("350",),
    "363": ("359", "360", "361"), "364": ("359", "362"), "365": ("359", "362"), "366": ("359", "362"),
    "376": ("375",),
    "379": ("378",), "380": ("378",), "381": ("378",), "382": ("378",),
    "384": ("383",), "385": ("383",), "386": ("383",), "387": ("383",),
    "392": ("390",), "393": ("390",), "394": ("390",), "397": ("390",),
    "395": ("391",), "396": ("391",), "399": ("391",), "402": ("391",),
    "406": ("405",), "407": ("405",), "408": ("405",), "409": ("405",),
    "417": ("415",), "418": ("415",), "419": ("415",), "420": ("415",),
    "427": ("425",), "435": ("425",), "436": ("425",), "448": ("442",), "452": ("442",),
    "465": ("463",), "466": ("463",), "467": ("463", "464"), "468": ("463", "464"), "471": ("464",),
    "500": ("499",), "506": ("503",),
}
DEFINITION_OF = {
    "107": "abetment", "120A": "criminal conspiracy", "146": "rioting", "299": "culpable homicide", "300": "murder",
    "319": "hurt", "320": "grievous hurt", "339": "wrongful restraint", "340": "wrongful confinement",
    "350": "criminal force", "359": "kidnapping", "360": "kidnapping from India",
    "361": "kidnapping from lawful guardianship", "362": "abduction", "375": "rape", "378": "theft",
    "383": "extortion", "390": "robbery", "391": "dacoity", "405": "criminal breach of trust", "415": "cheating",
    "425": "mischief", "442": "house-trespass", "463": "forgery", "464": "making a false document",
    "499": "defamation", "503": "criminal intimidation",
}

STOP = set("""a an the of and or to in on for by with at from as is are was were be been being that this these those
it its which who whom whose what when where why how under section sections act code court case cases accused
appellant respondent state learned said also any such not no shall may can would could should has have had do does
did there their them they he she his her him we our you your i""".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


@dataclass
class Hit:
    pid: str
    case_id: str
    opinion: int
    para_no: int
    title: str
    year: int | None
    text: str
    score: float
    why: list[str] = field(default_factory=list)
    cited_from: str | None = None  # for precedent hits: the retrieved paragraph that cites this case

    @property
    def label(self) -> str:
        op = "" if self.opinion == 0 else f" (opinion {self.opinion})"
        return f"{self.case_id} ¶{self.para_no}{op}"


class Retriever:
    def __init__(self):
        self.chunks = pd.read_parquet(f"{RAG}/chunks.parquet")
        self.emb = np.load(f"{RAG}/emb.npy").astype(np.float32)
        paras = pd.read_parquet("data/graph/nodes_paragraph.parquet")
        self.para_text = dict(zip(paras["pid"], paras["text"]))
        cases = pd.read_parquet("data/graph/nodes_case.parquet")
        corpus = cases[cases["in_corpus"]]
        self.titles = dict(zip(corpus["case_id"], corpus["title"].fillna("")))
        self.years = dict(zip(corpus["case_id"], corpus["year"]))
        self.chunk_rows = self.chunks.groupby("case_id").indices  # case_id -> chunk row numbers
        self.driver = driver()
        with self.driver.session() as s:
            self.pagerank = {r["id"]: r["pr"] or 0.0 for r in s.run(
                "MATCH (c:Case) WHERE c.pagerank IS NOT NULL RETURN c.case_id AS id, c.pagerank AS pr")}

    @cached_property
    def bm25(self):
        import bm25s
        index = bm25s.BM25()
        index.index([tokens(t) for t in self.chunks["text"]], show_progress=False)
        return index

    @cached_property
    def encoder(self):
        import torch
        from sentence_transformers import SentenceTransformer
        meta = dict(l.strip().split("=", 1) for l in open(f"{RAG}/embed_meta.txt"))
        return SentenceTransformer(meta["model"], device="mps" if torch.backends.mps.is_available() else "cpu")

    @cached_property
    def reranker(self):
        import torch
        from sentence_transformers import CrossEncoder
        mps = torch.backends.mps.is_available()
        ce = CrossEncoder(RERANKER, device="mps" if mps else "cpu", max_length=384)
        if mps:
            ce.model.half()
        return ce

    def embed_query(self, q: str) -> np.ndarray:
        # bge retrieval instruction for queries (passages are embedded without it)
        return self.encoder.encode("Represent this sentence for searching relevant passages: " + q,
                                   normalize_embeddings=True)

    # --- query analysis ----------------------------------------------------------------------------------------
    def provisions(self, q: str) -> tuple[list[str], list[str], list[str]]:
        """Provision pids named in the question plus their old <-> new counterparts (IPC/BNS, CrPC/BNSS, IEA/BSA),
        readable correspondence lines for the prompt ('Indian Penal Code, 1860 s.302 = Bharatiya Nyaya Sanhita,
        2023 s.103(1)'), and lines for the IPC sections that define the offences those provisions punish
        ('Indian Penal Code, 1860 s.300 (murder) = Bharatiya Nyaya Sanhita, 2023 s.101'). Definitions are context
        for the answer only; they are not used to rank paragraphs (rag_bench: boosting provisions the question did
        not name lowers MRR)."""
        from link.provisions import extract
        pids = {f"{r['act']}|{r['section']}" for r in extract(q, {}) if r["act"]}
        if not pids:
            return [], [], []
        found, lines = set(pids), []
        with self.driver.session() as s:
            rows = s.run("""UNWIND $pids AS pid MATCH (p:Provision {pid: pid})-[e:MAPS_TO]-(o:Provision)
                            RETURN startNode(e).pid AS old, endNode(e).pid AS new, e.new_sub AS sub""",
                         pids=sorted(pids))
            for r in rows:
                found.update([r["old"], r["new"]])
                lines.append(self._corr_line(r["old"], r["new"], r["sub"] or ""))
        ipc = sorted({p.split("|")[1] for p in found if p.startswith(IPC + "|")})
        related = []
        for sec in ipc:
            for d in DEFINED_BY.get(sec, ()):
                if f"{IPC}|{d}" in found:
                    continue
                for bns, sub in self.ipc_bns.get(d, []):
                    line = self._corr_line(f"{IPC}|{d}", f"{BNS}|{bns}", sub, DEFINITION_OF.get(d))
                    if line not in related:
                        related.append(line)
        return sorted(found), list(dict.fromkeys(lines)), related

    @staticmethod
    def _corr_line(old: str, new: str, sub: str, label: str | None = None) -> str:
        o, n = (x.replace("|", " s.") for x in (old, new))
        if label:
            o += f" ({label})"
        return f"{o} = {n}{'' if not sub or sub.startswith('(') else ' '}{sub}"

    @cached_property
    def ipc_bns(self) -> dict:
        """IPC section -> [(BNS section, sub-section)] from the reviewed correspondence table."""
        m = pd.read_csv("data/ref/ipc_bns.csv", dtype=str).fillna("")
        out: dict = {}
        for r in m.itertuples():
            out.setdefault(r.ipc, []).append((r.bns, r.bns_sub))
        return out

    @cached_property
    def all_titles(self) -> dict:
        """every Case node with a 'X v. Y' title, corpus or external (external = cited-only precedents)."""
        cases = pd.read_parquet("data/graph/nodes_case.parquet")
        t = cases.dropna(subset=["title"])
        t = t[t["title"].str.contains(r"\sv(?:s)?\.?\s|versus", case=False, regex=True)]
        return dict(zip(t["case_id"], t["title"]))

    @cached_property
    def lawyer_keys(self) -> set[str]:
        try:
            return set(pd.read_parquet("data/graph/nodes_lawyer.parquet")["key"])
        except FileNotFoundError:  # graph built without the NER pass
            return set()

    def named_cases(self, q: str) -> list[str]:
        """Cases the question names ('... in Sharad Birdhichand Sarda v. State of Maharashtra ...' or by neutral
        citation '2025 INSC 529'), corpus or external, best match first. Matches the 'X v. Y' span of the question
        against every case title."""
        cited = [c for c in scoping.neutral_citations(q) if c in self.titles]
        m = re.search(r"((?:[A-Z][\w.@&'-]*\s+){1,6}(?:v\.?|vs\.?|versus)\s+(?:(?:the|of|and)\s+|[A-Z][\w.()&'-]*\s*){1,8})", q)
        if not m:
            return cited
        from rapidfuzz import fuzz, process
        span = m.group(1).strip(" ,.")
        hits = process.extract(span, self.all_titles, scorer=fuzz.token_set_ratio, limit=5, score_cutoff=88)
        top = hits[0][1] if hits else 0
        return cited + [cid for _, sc, cid in hits if sc >= top - 3 and cid not in cited]

    def citing_paragraphs(self, case_ids: list[str]) -> dict[str, str]:
        """paragraph pid -> cited case id, for corpus paragraphs that cite any of the given cases."""
        if not case_ids:
            return {}
        with self.driver.session() as s:
            rows = s.run("""UNWIND $ids AS id MATCH (a:Case)-[e:CITES]->(b:Case {case_id: id})
                            RETURN a.case_id AS src, e.paras AS paras, b.case_id AS dst""", ids=case_ids)
            out = {}
            for r in rows:
                for p in r["paras"]:
                    op, no = p.split(":")
                    out.setdefault(f"{r['src']}/{op}/{no}", r["dst"])
        return out

    # --- text search --------------------------------------------------------------------------------------------
    def text_search(self, q: str, qv: np.ndarray, use_bm25: bool = True, use_dense: bool = True,
                    cases: set[str] | None = None, drop: set[str] = frozenset()) -> pd.Series:
        """chunk row -> fused RRF score (BM25 and / or dense), over the chunks of `cases` only when given; `drop`
        removes words from the BM25 query (scope words, rag/scope.py)."""
        scores: dict[int, float] = {}
        mask = self.chunks["case_id"].isin(cases).to_numpy() if cases is not None else None
        if use_bm25:
            q_tokens = [t for t in tokens(q) if t not in drop] or tokens(q)
            res, sc = self.bm25.retrieve([q_tokens], k=K_TEXT, show_progress=False,
                                         weight_mask=None if mask is None else mask.astype(np.float32))
            for rank, (i, v) in enumerate(zip(res[0], sc[0])):
                if mask is None or (mask[int(i)] and v > 0):
                    scores[int(i)] = scores.get(int(i), 0) + 1 / (RRF_K + rank)
        if use_dense:
            dense = self.emb @ qv
            if mask is not None:
                dense = np.where(mask, dense, -np.inf)
            top = np.argpartition(-dense, K_TEXT)[:K_TEXT]
            for rank, i in enumerate(top[np.argsort(-dense[top])]):
                scores[int(i)] = scores.get(int(i), 0) + W_DENSE / (RRF_K + rank)
        return pd.Series(scores, dtype=float)

    # --- graph --------------------------------------------------------------------------------------------------
    def provisions_named(self, q: str) -> list[str]:
        from link.provisions import extract
        return sorted({f"{r['act']}|{r['section']}" for r in extract(q, {}) if r["act"]})

    def provision_hits(self, pids: list[str]) -> tuple[set[str], set[str]]:
        """(paragraph pids invoking any of the provisions, case ids invoking any of them)."""
        if not pids:
            return set(), set()
        with self.driver.session() as s:
            rows = s.run("""UNWIND $pids AS pid MATCH (c:Case)-[e:INVOKES]->(:Provision {pid: pid})
                            RETURN c.case_id AS case_id, e.paras AS paras""", pids=pids)
            paras, cases = set(), set()
            for r in rows:
                cases.add(r["case_id"])
                for p in r["paras"]:
                    op, no = p.split(":")
                    paras.add(f"{r['case_id']}/{op}/{no}")
        return paras, cases

    def cited_from(self, seeds: list[Hit]) -> list[tuple[str, Hit]]:
        """(cited corpus case id, citing seed hit), citations made in the seed paragraph itself, most-cited first."""
        refs = [{"case_id": h.case_id, "para": f"{h.opinion}:{h.para_no}", "pid": h.pid} for h in seeds]
        with self.driver.session() as s:
            rows = s.run("""UNWIND $refs AS ref
                            MATCH (a:Case {case_id: ref.case_id})-[e:CITES]->(b:Case)
                            WHERE ref.para IN e.paras AND NOT b.case_id STARTS WITH 'ext:'
                            RETURN b.case_id AS dst, ref.pid AS src, COUNT { (b)<-[:CITES]-() } AS cited_by
                            ORDER BY cited_by DESC""", refs=refs)
            by_pid = {h.pid: h for h in seeds}
            out, seen = [], set()
            for r in rows:
                if r["dst"] not in seen:
                    seen.add(r["dst"])
                    out.append((r["dst"], by_pid[r["src"]]))
        return out

    # --- assembly -----------------------------------------------------------------------------------------------
    def _hit(self, pid: str, score: float, why: list[str], chunk_rows: list[int]) -> Hit:
        c = self.chunks.iloc[chunk_rows[0]]
        full = self.para_text.get(pid, "")
        if len(full.split()) <= MAX_WORDS:
            text = full
        else:  # long paragraph: the matched windows only, in document order
            parts = self.chunks.iloc[sorted(chunk_rows)]["text"].tolist()
            text = " … ".join(parts)
            text = " ".join(text.split()[:MAX_WORDS]) + " …"
        return Hit(pid=pid, case_id=c["case_id"], opinion=int(c["opinion"]), para_no=int(c["para_no"]),
                   title=self.titles.get(c["case_id"], ""), year=self.years.get(c["case_id"]),
                   text=text, score=score, why=why)

    def search(self, q: str, top: int = TOP_PARAS, precedents: int = TOP_PRECEDENTS, per_case: int = PER_CASE,
               use_bm25: bool = True, use_dense: bool = False, use_graph: bool = True, rerank: bool = False,
               use_scope: bool = True) -> dict:
        """use_bm25 / use_dense / use_graph / use_scope switch components off for ablations (eval/rag_bench.py,
        eval/scope_bench.py); rerank orders the top RERANK_DEPTH paragraphs with a cross-encoder before the per-case
        cap."""
        qv = self.embed_query(q)
        prov, prov_map, related = self.provisions(q) if use_graph else ([], [], [])
        named_prov = set(self.provisions_named(q)) if use_graph else set()
        # bridge one way only: a question in new-Act terms (BNSS 528) needs the judgments citing the old section
        # (CrPC 482); the reverse would boost the few recent judgments over the better older matches
        bridge = [p for p in prov if p not in named_prov and p.split("|")[0] in OLD_ACTS]
        prov_paras, prov_cases = self.provision_hits(bridge)
        named = self.named_cases(q) if use_graph else []
        scope = scoping.Scope()
        if use_graph and use_scope:
            with self.driver.session() as s:
                scope = scoping.analyse(q, s, self.lawyer_keys, [c for c in named if c in self.titles])
        allowed = scope.case_ids
        if allowed is not None and len(allowed) == 1:
            per_case = top  # a question about one judgment reads as much of it as it needs

        fused = self.text_search(q, qv, use_bm25, use_dense, allowed, scope.drop_words if allowed is not None else set())
        rows = self.chunks.loc[fused.index, ["pid", "case_id"]].assign(score=fused.values, row=fused.index)
        # named cases: their own best chunks (corpus cases), and the paragraphs that cite them (corpus or external:
        # an old precedent outside the corpus is known through how later judgments discuss it)
        citing = self.citing_paragraphs(named)
        pid_rows = self.chunks.groupby("pid").indices if citing else {}
        extra_rows = []
        for cid in named:
            idx = self.chunk_rows.get(cid, [])
            if len(idx):
                extra_rows += list(idx[np.argsort(-(self.emb[idx] @ qv))[:3]])
        for pid in citing:
            extra_rows += list(pid_rows.get(pid, []))
        if scope.paras:
            pid_rows = pid_rows or self.chunks.groupby("pid").indices
            for pid in scope.paras:
                extra_rows += list(pid_rows.get(pid, []))
        if allowed is not None:
            extra_rows = [i for i in extra_rows if self.chunks.at[int(i), "case_id"] in allowed]
        if extra_rows:
            extra_rows = np.array(sorted(set(extra_rows) - set(rows["row"])), dtype=int)
            sims = self.emb[extra_rows] @ qv  # rank the added chunks among themselves by similarity, RRF-style
            ranked = extra_rows[np.argsort(-sims)][:K_TEXT]
            extra = self.chunks.loc[ranked, ["pid", "case_id"]].assign(
                score=[1 / (RRF_K + i) for i in range(len(ranked))], row=ranked)
            rows = pd.concat([rows, extra])

        para = rows.groupby("pid").agg(case_id=("case_id", "first"), score=("score", "max"),
                                       rows=("row", list)).reset_index()
        why = {pid: ["text"] for pid in para["pid"]}
        for i, r in para.iterrows():
            bonus = 0.0
            if r["pid"] in prov_paras:
                bonus += W_PARA_PROV
                why[r["pid"]].append("invokes provision")
            elif r["case_id"] in prov_cases:
                bonus += W_CASE_PROV
                why[r["pid"]].append("case invokes provision")
            if r["case_id"] in named:
                bonus += W_NAMED_CASE
                why[r["pid"]].append("named case")
            elif r["pid"] in citing:
                bonus += W_NAMED_CASE
                why[r["pid"]].append("cites named case")
            if r["pid"] in scope.paras:
                bonus += W_WITNESS
                why[r["pid"]].append(f"names {scope.paras[r['pid']]}")
            if use_graph:
                bonus += W_PAGERANK * math.log1p(self.pagerank.get(r["case_id"], 0.0))
            para.at[i, "score"] = r["score"] + bonus
        para = para.sort_values("score", ascending=False)
        if rerank and len(para):
            cand = para.head(RERANK_DEPTH)
            cand_hits = [self._hit(r.pid, r.score, why[r.pid], r.rows) for r in cand.itertuples()]
            ce = self.reranker.predict([(q, h.text) for h in cand_hits], batch_size=16)
            order = np.argsort(-np.asarray(ce))
            hits, seen = [], {}
            for i in order:
                h = cand_hits[i]
                if seen.get(h.case_id, 0) < per_case:
                    seen[h.case_id] = seen.get(h.case_id, 0) + 1
                    h.score = float(ce[i])
                    h.why.append("reranked")
                    hits.append(h)
            hits = hits[:top]
        else:
            para = para[para.groupby("case_id").cumcount() < per_case].head(top)
            hits = [self._hit(r.pid, r.score, why[r.pid], r.rows) for r in para.itertuples()]

        # precedents cited from the retrieved paragraphs, best paragraph of each by similarity to the question
        prec, have = [], {h.case_id for h in hits}
        for dst, src in (self.cited_from(hits) if precedents and use_graph and allowed is None else []):
            if len(prec) >= precedents or dst in have:
                continue
            idx = self.chunk_rows.get(dst)
            if idx is None or not len(idx):
                continue
            sims = self.emb[idx] @ qv
            row = int(idx[int(np.argmax(sims))])
            pid = self.chunks.at[row, "pid"]
            rows_same = [int(i) for i in idx if self.chunks.at[int(i), "pid"] == pid]
            h = self._hit(pid, float(sims.max()), ["cited precedent"], rows_same)
            h.cited_from = src.label
            prec.append(h)
            have.add(dst)
        return {"question": q, "provisions": prov, "provision_map": prov_map, "related_map": related,
                "named_cases": named, "scope": scope.filters, "scope_notes": scope.notes, "hits": hits,
                "precedents": prec}


def main():
    q = " ".join(sys.argv[1:]) or "When can conviction rest solely on circumstantial evidence?"
    r = Retriever().search(q)
    print("provisions:", r["provisions"], "| named cases:", r["named_cases"], "| scope:", r["scope"])
    for h in r["hits"] + r["precedents"]:
        tag = f"cited from {h.cited_from}" if h.cited_from else ", ".join(h.why)
        print(f"{h.score:.4f}  {h.label:28}  {h.title[:50]:50}  [{tag}]")
        print("        ", h.text[:160].replace("\n", " "))


if __name__ == "__main__":
    main()
