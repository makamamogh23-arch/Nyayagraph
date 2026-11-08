"""Per-entity NER P/R/F1, strict (exact span+label) and lenient (any overlap + same label, 1:1 matching).

Gold jsonl: rows with id, text|input, spans|gold_spans  (e.g. data/instruct/test.jsonl).
Pred jsonl: rows with id and ONE of
  spans:    [{start, end, label}]          (encoder / already aligned)
  entities: [{label, text}]                (LLM, parsed)  -> aligned to char spans
  output:   raw LLM string                 -> parsed + aligned
Missing pred ids count as empty predictions.

Usage: python -m eval.ner_f1 GOLD PRED [--json out.json]
"""
import argparse
import json
from collections import defaultdict

from ner.common import align, parse_llm_output, read_jsonl


def pred_spans(row: dict, text: str) -> list[dict]:
    if "spans" in row:
        return row["spans"]
    ents = row["entities"] if "entities" in row else parse_llm_output(row.get("output", ""))
    return align(text, ents)


def match(gold: list[dict], pred: list[dict], lenient: bool) -> list[tuple]:
    """Return ('tp'|'fp'|'fn', label) events."""
    exact = lambda p, g: (g["start"], g["end"]) == (p["start"], p["end"])
    overlap = lambda p, g: p["start"] < g["end"] and g["start"] < p["end"]
    used, matched = set(), set()
    # exact matches first so a loose-boundary pred can't steal a gold that has an exact pred
    for test in (exact, overlap) if lenient else (exact,):
        for j, p in enumerate(pred):
            if j in matched:
                continue
            for i, g in enumerate(gold):
                if i not in used and g["label"] == p["label"] and test(p, g):
                    used.add(i)
                    matched.add(j)
                    break
    return ([("tp" if j in matched else "fp", p["label"]) for j, p in enumerate(pred)]
            + [("fn", g["label"]) for i, g in enumerate(gold) if i not in used])


def prf(c: dict) -> dict:
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": p, "r": r, "f1": f, "support": c["tp"] + c["fn"]}


def score(gold_rows: list[dict], pred_rows: list[dict]) -> dict:
    preds = {str(r["id"]): r for r in pred_rows}
    counts = {m: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}) for m in ("strict", "lenient")}
    for g in gold_rows:
        text = g.get("text", g.get("input"))
        gold = g.get("spans", g.get("gold_spans"))
        pr = pred_spans(preds[str(g["id"])], text) if str(g["id"]) in preds else []
        for mode in counts:
            for kind, label in match(gold, pr, lenient=(mode == "lenient")):
                counts[mode][label][kind] += 1
    out = {}
    for mode, per in counts.items():
        labels = {l: prf(c) for l, c in sorted(per.items())}
        total = {k: sum(c[k] for c in per.values()) for k in ("tp", "fp", "fn")}
        gold_labels = [l for l in labels if labels[l]["support"]]
        out[mode] = {
            "per_label": labels,
            "micro": prf(total),
            "macro_f1": sum(labels[l]["f1"] for l in gold_labels) / max(len(gold_labels), 1),
        }
    out["coverage"] = sum(str(g["id"]) in preds for g in gold_rows) / max(len(gold_rows), 1)
    return out


def print_report(res: dict):
    labels = sorted(set(res["strict"]["per_label"]) | set(res["lenient"]["per_label"]))
    print(f"{'label':<14}{'supp':>6}  {'strict P/R/F1':>20}  {'lenient P/R/F1':>20}")
    fmt = lambda d: f"{d['p']:.3f}/{d['r']:.3f}/{d['f1']:.3f}"
    zero = {"p": 0, "r": 0, "f1": 0, "support": 0}
    for l in labels:
        s, n = res["strict"]["per_label"].get(l, zero), res["lenient"]["per_label"].get(l, zero)
        print(f"{l:<14}{s['support']:>6}  {fmt(s):>20}  {fmt(n):>20}")
    print(f"{'MICRO':<14}{res['strict']['micro']['support']:>6}  {fmt(res['strict']['micro']):>20}  "
          f"{fmt(res['lenient']['micro']):>20}")
    print(f"{'MACRO F1':<14}{'':>6}  {res['strict']['macro_f1']:>20.3f}  {res['lenient']['macro_f1']:>20.3f}")
    print(f"pred coverage: {res['coverage']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("pred")
    ap.add_argument("--json")
    args = ap.parse_args()
    res = score(read_jsonl(args.gold), read_jsonl(args.pred))
    print_report(res)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
