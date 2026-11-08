"""Retrieval benchmark for rag/retrieve.py, built from the SCR editors' headnotes.

Gold: each "HELD" point in a headnote states a proposition and cites the judgment paragraphs it comes from
("1.3 ... evidence of a child witness must find adequate corroboration ... [Paras 22, 23]"). Headnotes are not in
the retrieval index, so the editor's summary is an independent query whose answer is known.

  build  parse HELD points -> sample (seeded, <= 2 per case) -> Claude Haiku labels each as a general legal
         proposition or case-specific facts, and rewrites propositions as a short research question with no party
         names, dates or copied phrasing (cached in data/bench/haiku.jsonl). Split dev / test by case.
  run    score retrieval configurations on two query styles:
           held      the editor's text (first 80 words): close to the judgment's own wording (favours BM25)
           question  the rewritten question: how a lawyer would ask
           newact    questions naming IPC / CrPC / Evidence Act sections, rewritten to the BNS / BNSS / BSA numbers
                     (needs the MAPS_TO bridge: the judgments cite the old sections)
         Metrics over the ranked paragraph list: para@k (a gold paragraph in top k), case@k (any paragraph of the
         gold case in top k), MRR (first gold paragraph). Only the source case counts as correct although other
         judgments often state the same rule, so absolute numbers are a lower bound; the benchmark compares setups.

Usage:
  python -m eval.rag_bench build [--n 600]
  python -m eval.rag_bench run [--split test] [--configs full,hybrid,bm25,dense]
"""
import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

B = "data/bench"
HAIKU = "claude-haiku-5-5"
REF = re.compile(r"[\[\(]\s*Para(?:graph)?s?\.?\s*([\d,\s\-–&and]+?)\s*[\]\)]", re.I)
ITEM = re.compile(r"(?m)^\s*(\d{1,2}(?:\.\d{1,2})?)\.?\s+(?=[A-Z])")
KS = (1, 5, 10)


# --- build -----------------------------------------------------------------------------------------------------

def held_points() -> pd.DataFrame:
    cases = pd.read_parquet("data/processed/cases.parquet").dropna(subset=["headnote"])
    have = set(pd.read_parquet("data/graph/nodes_paragraph.parquet")["pid"])
    rows = []
    for r in cases.itertuples():
        i = r.headnote.find("HELD")
        if i < 0:
            continue
        held = r.headnote[i + 5:]
        starts = [m.start() for m in ITEM.finditer(held)] or [0]
        for a, b in zip(starts, starts[1:] + [len(held)]):
            seg = held[a:b]
            m = REF.search(seg)
            if not m:
                continue
            text = ITEM.sub("", seg[:m.start()], count=1)
            text = re.sub(r"\s+", " ", re.sub(r"\s*-\s*\n\s*", "", text)).strip()
            nums = []
            for part in re.split(r",|&|and", m.group(1)):
                rg = [int(x) for x in re.findall(r"\d+", part)]
                nums += list(range(rg[0], rg[1] + 1)) if len(rg) == 2 and 0 < rg[1] - rg[0] <= 6 else rg
            gold = sorted({f"{r.case_id}/0/{n}" for n in nums if f"{r.case_id}/0/{n}" in have})
            if len(text.split()) >= 12 and gold:
                rows.append({"case_id": r.case_id, "held": text, "gold": gold})
    return pd.DataFrame(rows)


def label(points: pd.DataFrame) -> pd.DataFrame:
    import anthropic
    from pydantic import BaseModel, Field

    from rag.ask import load_key

    class Label(BaseModel):
        kind: str = Field(description="'proposition' if it states a rule or principle of law that other cases could "
                                      "also apply; 'facts' if it is mainly about this case's facts or orders")
        question: str = Field(description="for a proposition: a research question of at most 30 words that this "
                                          "holding answers; no party names, dates, places or case-specific facts, "
                                          "and do not copy the wording; empty for facts")

    load_key()
    client = anthropic.Anthropic()
    path = f"{B}/haiku.jsonl"
    cache = {}
    if os.path.exists(path):
        cache = {r["key"]: r for r in map(json.loads, open(path))}

    def key(t):
        return hashlib.sha1(f"{HAIKU}\n{t}".encode()).hexdigest()[:16]

    def one(t):
        r = client.messages.parse(
            model=HAIKU, max_tokens=4096, output_config={"effort": "low"}, output_format=Label,
            system="You prepare evaluation queries for a legal research system over Indian Supreme Court criminal "
                   "judgments. You see one 'held' point written by a law reporter.",
            messages=[{"role": "user", "content": t}])
        p = r.parsed_output
        return {"key": key(t), "kind": (p.kind if p else "facts").strip().lower(),
                "question": (p.question if p else "").strip(), "in": r.usage.input_tokens, "out": r.usage.output_tokens}

    todo = [t for t in points["held"] if key(t) not in cache]
    with open(path, "a") as f, ThreadPoolExecutor(8) as ex:
        for rec in ex.map(one, todo):
            f.write(json.dumps(rec) + "\n")
            cache[rec["key"]] = rec
    if todo:
        tin = sum(cache[key(t)]["in"] for t in todo)
        tout = sum(cache[key(t)]["out"] for t in todo)
        print(f"haiku calls {len(todo)}  tokens {tin:,} / {tout:,}  ${(tin * 0.10 + tout * 0.50) / 1e6:.3f}")
    lab = points["held"].map(lambda t: cache[key(t)])
    return points.assign(kind=lab.map(lambda r: r["kind"]), question=lab.map(lambda r: r["question"]))


def build(n: int, seed: int = 0) -> None:
    os.makedirs(B, exist_ok=True)
    pts = held_points()
    pts = pts.sample(frac=1, random_state=seed).groupby("case_id").head(2).head(n).reset_index(drop=True)
    pts = label(pts)
    pts["split"] = pts["case_id"].map(lambda c: "dev" if int(hashlib.md5(c.encode()).hexdigest(), 16) % 5 < 2 else "test")
    pts.to_parquet(f"{B}/bench.parquet", index=False)
    props = pts[(pts["kind"] == "proposition") & (pts["question"].str.len() > 0)]
    print(f"held points: {len(held_points()):,} parsed; sampled {len(pts)} from {pts['case_id'].nunique()} cases; "
          f"propositions {len(props)} (dev {int((props['split'] == 'dev').sum())}, test {int((props['split'] == 'test').sum())})")
    for r in props.head(3).itertuples():
        print(f"  held: {r.held[:140]}...\n  question: {r.question}\n  gold: {r.gold}")


# --- run -------------------------------------------------------------------------------------------------------

NEW_SHORT = {"ipc_bns": ("Indian Penal Code, 1860", "BNS"), "crpc_bnss": ("Code of Criminal Procedure, 1973", "BNSS"),
             "iea_bsa": ("Indian Evidence Act, 1872", "BSA")}


def new_act_query(q: str, maps: dict) -> str | None:
    """Rewrite every old-Act section the question names to its new-Act counterpart ('Section 405 IPC' ->
    'Section 316 of the BNS'); None if the question names none or one has no single counterpart."""
    from link.provisions import extract
    hits = extract(q, {})
    if not hits:
        return None
    out, off = q, 0
    for span in sorted({(x["start"], x["end"]) for x in hits}):
        tgt = [maps.get((x["act"], x["section"])) for x in hits if (x["start"], x["end"]) == span]
        if not all(tgt) or len({t[0] for t in tgt}) != 1:
            return None
        rep = ("Sections " if len(tgt) > 1 else "Section ") + " and ".join(t[1] for t in tgt) + " of the " + tgt[0][0]
        out = out[:span[0] + off] + rep + out[span[1] + off:]
        off += len(rep) - (span[1] - span[0])
    return out if out != q else None


def new_act_maps() -> dict:
    maps = {}
    for csv, (old_act, short) in NEW_SHORT.items():
        m = pd.read_csv(f"data/ref/{csv}.csv", dtype=str).fillna("")
        o, n = ("ipc", "bns") if csv == "ipc_bns" else ("old", "new")
        for sec, g in m.groupby(o):
            if g[n].nunique() == 1:
                maps[(old_act, sec)] = (short, g[n].iloc[0])
    return maps


CONFIGS = {  # name -> retriever.search keyword arguments
    "bm25": {"use_bm25": True, "use_dense": False, "use_graph": False},
    "dense": {"use_bm25": False, "use_dense": True, "use_graph": False},
    "hybrid": {"use_bm25": True, "use_dense": True, "use_graph": False},
    "full": {"use_bm25": True, "use_dense": True, "use_graph": True},  # deployed before the benchmark
    "bm25+rerank": {"use_bm25": True, "use_dense": False, "use_graph": False, "rerank": True},
    "bm25+graph+rerank": {"use_bm25": True, "use_dense": False, "use_graph": True, "rerank": True},
    "bm25+graph": {"use_bm25": True, "use_dense": False, "use_graph": True},  # default after the benchmark
}


def score(ranked: list[tuple[str, str]], gold: list[str]) -> dict:
    """ranked = [(pid, case_id)] in order."""
    gold_set, gold_case = set(gold), gold[0].split("/")[0]
    first = next((i for i, (p, _) in enumerate(ranked, 1) if p in gold_set), None)
    out = {"mrr": 1 / first if first else 0.0}
    for k in KS:
        out[f"para@{k}"] = float(first is not None and first <= k)
        out[f"case@{k}"] = float(any(c == gold_case for _, c in ranked[:k]))
    return out


def run(split: str, configs: list[str], styles: list[str]) -> pd.DataFrame:
    from rag.retrieve import Retriever

    bench = pd.read_parquet(f"{B}/bench.parquet")
    bench = bench[(bench["kind"] == "proposition") & (bench["question"].str.len() > 0)]
    if split != "all":
        bench = bench[bench["split"] == split]
    r = Retriever()
    rows = []
    bench = bench.reset_index(drop=True)
    maps = new_act_maps() if "newact" in styles else {}
    for style in styles:
        if style == "held":
            queries = bench["held"].map(lambda t: " ".join(t.split()[:80]))
        elif style == "newact":  # questions naming old sections, rephrased with the new Acts' numbers
            queries = bench["question"].map(lambda q: new_act_query(q, maps))
        else:
            queries = bench["question"]
        for cfg in configs:
            for q, gold in zip(queries, bench["gold"]):
                if not isinstance(q, str):
                    continue
                res = r.search(q, top=max(KS), precedents=0, per_case=max(KS), **CONFIGS[cfg])
                ranked = [(h.pid, h.case_id) for h in res["hits"]]
                rows.append({"style": style, "config": cfg, "case_id": gold[0].split("/")[0], "qid": len(rows),
                             **score(ranked, list(gold))})
    df = pd.DataFrame(rows)
    df.to_parquet(f"{B}/per_query_{split}.parquet", index=False)
    table = df.drop(columns=["case_id", "qid"]).groupby(["style", "config"], sort=False).mean().round(3)
    table.insert(0, "n", df.groupby(["style", "config"], sort=False).size())
    return table


def paired_bootstrap(split: str, a: str, b: str, metric: str, style: str = "question", n: int = 2000, seed: int = 0):
    """mean(b - a) on `metric` with a 95% CI, resampling cases (queries from one case stay together)."""
    import numpy as np
    df = pd.read_parquet(f"{B}/per_query_{split}.parquet")
    df = df[df["style"] == style]
    x = df[df["config"] == a].reset_index(drop=True)
    y = df[df["config"] == b].reset_index(drop=True)
    d = pd.DataFrame({"case": x["case_id"], "diff": y[metric].values - x[metric].values})
    per_case = d.groupby("case")["diff"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_case), size=(n, len(per_case)))
    s, c = per_case["sum"].to_numpy()[idx].sum(1), per_case["size"].to_numpy()[idx].sum(1)
    boots = s / c
    return d["diff"].mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=600)
    rr = sub.add_parser("run")
    rr.add_argument("--split", default="test", choices=["dev", "test", "all"])
    rr.add_argument("--configs", default="bm25,dense,hybrid,full")
    rr.add_argument("--styles", default="held,question")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.n)
    else:
        table = run(args.split, args.configs.split(","), args.styles.split(","))
        print(table.to_string())
        out = f"{B}/results_{args.split}.csv"
        table.to_csv(out)
        print("->", out)
        cfgs = args.configs.split(",")
        for style in [x for x in args.styles.split(",") if x != "held"]:
            for base in [c for c in ("bm25", "full") if c in cfgs]:
                for other in [c for c in cfgs if c != base and not (base == "full" and c == "bm25")]:
                    for metric in ("mrr", "para@10", "case@10"):
                        m, lo, hi = paired_bootstrap(args.split, base, other, metric, style)
                        print(f"  {style:8}  {other} - {base}  {metric:8} {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
