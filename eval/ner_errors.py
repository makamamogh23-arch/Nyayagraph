"""NER error taxonomy per label (gold-side + pred-side), to explain strict vs lenient gaps.

Gold entity outcome:  exact | boundary (same label, overlaps, not exact) | split (>=2 same-label preds overlap it)
                      | type (only overlapped by other labels) | missed
Pred outcome:         exact | boundary | fragment (one of >=2 same-label preds on one gold) | type | spurious

Usage: python -m eval.ner_errors GOLD PRED [--labels ORG PRECEDENT] [--examples 3]
"""
import argparse
from collections import Counter, defaultdict

from eval.ner_f1 import pred_spans
from ner.common import read_jsonl


def ov(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("pred")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    preds = {str(r["id"]): r for r in read_jsonl(args.pred)}
    g_out, p_out, confusion, examples = defaultdict(Counter), defaultdict(Counter), Counter(), defaultdict(list)
    for g in read_jsonl(args.gold):
        text, gold = g["input"], g["gold_spans"]
        pr = pred_spans(preds[str(g["id"])], text) if str(g["id"]) in preds else []
        for x in gold:
            same = [p for p in pr if p["label"] == x["label"] and ov(p, x)]
            other = [p for p in pr if p["label"] != x["label"] and ov(p, x)]
            if any((p["start"], p["end"]) == (x["start"], x["end"]) for p in same):
                k = "exact"
            elif len(same) >= 2:
                k = "split"
            elif same:
                k = "boundary"
            elif other:
                k = "type"
                confusion[(x["label"], other[0]["label"])] += 1
            else:
                k = "missed"
            g_out[x["label"]][k] += 1
            if k != "exact" and len(examples[(x["label"], k)]) < args.examples:
                got = " | ".join(f"{text[p['start']:p['end']]!r}:{p['label']}" for p in same + other) or "-"
                examples[(x["label"], k)].append(f"gold {text[x['start']:x['end']]!r}  pred {got}")
        for p in pr:
            same = [x for x in gold if x["label"] == p["label"] and ov(p, x)]
            if any((p["start"], p["end"]) == (x["start"], x["end"]) for x in same):
                k = "exact"
            elif same and sum(1 for q in pr if q["label"] == p["label"] and ov(q, same[0])) >= 2:
                k = "fragment"
            elif same:
                k = "boundary"
            elif any(ov(p, x) for x in gold):
                k = "type"
            else:
                k = "spurious"
            p_out[p["label"]][k] += 1
            if k == "spurious" and len(examples[(p["label"], k)]) < args.examples:
                examples[(p["label"], k)].append(f"pred {text[p['start']:p['end']]!r}  (no gold)")

    labels = args.labels or sorted(g_out)
    gk, pk = ["exact", "boundary", "split", "type", "missed"], ["exact", "boundary", "fragment", "type", "spurious"]
    print(f"{'GOLD':<14}" + "".join(f"{k:>9}" for k in gk) + "    PRED" + "".join(f"{k:>9}" for k in pk))
    for l in labels:
        gt, pt = sum(g_out[l].values()) or 1, sum(p_out[l].values()) or 1
        print(f"{l:<14}" + "".join(f"{g_out[l][k] / gt:>9.3f}" for k in gk) + "        "
              + "".join(f"{p_out[l][k] / pt:>9.3f}" for k in pk))
    print("\ntop type confusions (gold -> pred):", ", ".join(f"{a}->{b} {n}" for (a, b), n in confusion.most_common(8)))
    for l in labels:
        for k in gk[1:] + ["spurious"]:
            for e in examples.get((l, k), []):
                print(f"  [{l} {k}] {e[:160]}")


if __name__ == "__main__":
    main()
