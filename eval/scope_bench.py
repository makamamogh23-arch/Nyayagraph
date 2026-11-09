"""Benchmark for question scope (rag/scope.py): does naming a lower court or a lawyer retrieve the right judgments?

10 questions, 5 naming a High Court and 5 naming counsel, each also naming a provision. Gold is defined from the
graph, with no hand labels:
  in scope  judgments appealed from that High Court (APPEAL_FROM) / in which that lawyer appeared (APPEARED_IN)
  gold      in-scope judgments that also invoke the named provision (INVOKES)
Gold is a proxy for relevance: a gold judgment invokes the provision somewhere, not necessarily in the retrieved
paragraph, and an in-scope judgment that discusses the topic without citing the section is not gold.

Each question runs twice through rag/retrieve.py (BM25 + graph, the deployed default): with scope and without.
Metrics over the top 10 paragraphs:
  in_scope@10  share of paragraphs from in-scope judgments
  gold@10      share of paragraphs from gold judgments
  cases@10     distinct gold judgments among them
  rr           reciprocal rank of the first gold paragraph
No model calls; the scope entities themselves come from the fine-tuned NER pass.

Usage: python -m eval.scope_bench [--out results/scope_bench.md]
"""
import argparse
import json
import time

import pandas as pd

from rag import scope as scoping
from rag.retrieve import Retriever

K = 10
IPC, CRPC, IEA = "Indian Penal Code, 1860", "Code of Criminal Procedure, 1973", "Indian Evidence Act, 1872"
PCA, NDPS = "Prevention of Corruption Act, 1988", "Narcotic Drugs and Psychotropic Substances Act, 1985"

QUESTIONS = [  # (kind, entity: court name or lawyer key, provision pid, question)
    ("court", "High Court of Judicature at Allahabad", f"{IPC}|304B",
     "In appeals from the Allahabad High Court, how has the Supreme Court applied the presumption of dowry death "
     "under Section 304B IPC?"),
    ("court", "High Court of Punjab and Haryana", f"{NDPS}|50",
     "In appeals from the Punjab and Haryana High Court, what follows from non-compliance with Section 50 of the "
     "NDPS Act during a personal search?"),
    ("court", "High Court of Judicature at Bombay", f"{IEA}|27",
     "In appeals from the Bombay High Court, when is a recovery under Section 27 of the Evidence Act reliable?"),
    ("court", "High Court of Judicature at Madras", f"{PCA}|13",
     "In appeals from the Madras High Court, what must be proved for criminal misconduct under Section 13 of the "
     "Prevention of Corruption Act?"),
    ("court", "High Court of Delhi", f"{CRPC}|439",
     "In appeals from the Delhi High Court, what factors govern the grant or cancellation of bail under Section 439 "
     "CrPC?"),
    ("lawyer", "sidharth luthra", f"{CRPC}|482",
     "In cases argued by Sidharth Luthra, when did the Court quash criminal proceedings under Section 482 CrPC?"),
    ("lawyer", "tushar mehta", f"{PCA}|13",
     "In cases where Tushar Mehta appeared, how was criminal misconduct under Section 13 of the Prevention of "
     "Corruption Act proved?"),
    ("lawyer", "r basant", f"{IPC}|307",
     "In cases argued by R. Basant, how did the Court assess intention for attempt to murder under Section 307 IPC?"),
    ("lawyer", "v n raghupathy", f"{IPC}|498A",
     "In cases where V. N. Raghupathy appeared as counsel, how did the Court treat cruelty under Section 498A IPC?"),
    ("lawyer", "mukesh kumar maroria", f"{IPC}|420",
     "In cases where Mukesh Kumar Maroria appeared, when did the Court find that cheating under Section 420 IPC "
     "was not made out?"),
]


def gold_sets() -> list[tuple[set, set]]:
    g = "data/graph"
    inv = pd.read_parquet(f"{g}/edges_invokes.parquet").groupby("pid")["case_id"].apply(set)
    af = pd.read_parquet(f"{g}/edges_appeal_from.parquet").groupby("court")["case_id"].apply(set)
    ap = pd.read_parquet(f"{g}/edges_appeared_in.parquet").groupby("lawyer")["case_id"].apply(set)
    out = []
    for kind, entity, prov, _ in QUESTIONS:
        scope = (af if kind == "court" else ap).get(entity, set())
        out.append((scope, scope & inv.get(prov, set())))
    return out


def metrics(hits, scope: set, gold: set) -> dict:
    top = hits[:K]
    first = next((i for i, h in enumerate(top, 1) if h.case_id in gold), None)
    return {"in_scope@10": sum(h.case_id in scope for h in top) / K,
            "gold@10": sum(h.case_id in gold for h in top) / K,
            "cases@10": len({h.case_id for h in top if h.case_id in gold}),
            "rr": 1 / first if first else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scope_bench.md")
    args = ap.parse_args()
    r = Retriever()
    golds = gold_sets()
    rows = []
    for (kind, entity, prov, q), (scope, gold) in zip(QUESTIONS, golds):
        detected = (scoping.detect_courts(q) if kind == "court" else scoping.detect_lawyers(q, r.lawyer_keys))
        for use_scope in (False, True):
            t = time.time()
            res = r.search(q, precedents=0, use_scope=use_scope)
            m = metrics(res["hits"], scope, gold)
            rows.append({"kind": kind, "entity": entity, "provision": prov.replace("|", " s."), "question": q,
                         "scope": "on" if use_scope else "off", "detected": entity in detected,
                         "n_scope": len(scope), "n_gold": len(gold), **m, "seconds": round(time.time() - t, 2),
                         "top10": [h.label for h in res["hits"][:K]]})
    df = pd.DataFrame(rows)
    df.drop(columns=["top10"]).to_csv(args.out.replace(".md", ".csv"), index=False)
    with open(args.out.replace(".md", ".jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    cols = ["in_scope@10", "gold@10", "cases@10", "rr"]
    summary = df.groupby(["kind", "scope"])[cols].mean().round(3)
    overall = df.groupby("scope")[cols].mean().round(3)
    wide = df.pivot_table(index=["kind", "entity", "provision", "n_scope", "n_gold"], columns="scope",
                          values=["gold@10", "cases@10", "rr"], sort=False)
    lines = ["# Scope benchmark: lower court and counsel named in the question", "",
             f"`python -m eval.scope_bench`: {len(QUESTIONS)} questions, retrieval only (BM25 + graph), "
             f"each run with scope off and on. Top {K} paragraphs. Gold = in-scope judgments that invoke the named "
             "provision (graph-defined, see the module docstring). No model calls.", "",
             f"Scope detected for {int(df[df.scope == 'on'].detected.sum())} of {len(QUESTIONS)} questions.", "",
             "## Summary (mean over questions)", "", "| questions | scope | " + " | ".join(cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for (kind, sc), v in summary.iterrows():
        lines.append(f"| {kind} | {sc} | " + " | ".join(f"{v[c]:.3f}" if c != "cases@10" else f"{v[c]:.1f}"
                                                         for c in cols) + " |")
    for sc, v in overall.iterrows():
        lines.append(f"| **all** | {sc} | " + " | ".join(f"{v[c]:.3f}" if c != "cases@10" else f"{v[c]:.1f}"
                                                         for c in cols) + " |")
    lines += ["", "## Per question", "",
              "| kind | named | provision | in scope | gold | gold@10 off → on | gold cases@10 off → on | RR off → on |",
              "|---|---|---|---|---|---|---|---|"]
    for (kind, entity, prov, ns, ng), v in wide.iterrows():
        lines.append(f"| {kind} | {entity.title() if kind == 'lawyer' else entity} | {prov} | {ns} | {ng} | "
                     f"{v[('gold@10', 'off')]:.1f} → {v[('gold@10', 'on')]:.1f} | "
                     f"{v[('cases@10', 'off')]:.0f} → {v[('cases@10', 'on')]:.0f} | "
                     f"{v[('rr', 'off')]:.2f} → {v[('rr', 'on')]:.2f} |")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", args.out, "+ .csv, .jsonl (top 10 per run)")


if __name__ == "__main__":
    main()
