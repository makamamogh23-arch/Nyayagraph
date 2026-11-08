"""Cost report for the Claude Haiku linking tie-break, from the per-call usage stored in the decision cache.

  python -m link.cost

Rates: Claude Haiku 5.5, prompts <= 100K tokens: $0.10 / MTok input, $0.50 / MTok output (output includes the
model's thinking tokens). Every tie-break prompt is well under 1K tokens, so the higher long-prompt card never applies.
"""
import json

import os

from link.resolve import CACHE, MODEL

SUPERSEDED = os.path.join(os.path.dirname(CACHE), "link_llm_superseded.jsonl")  # billed, then re-asked

RATE_IN, RATE_OUT = 0.10 / 1e6, 0.50 / 1e6


def main():
    with open(CACHE) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    old = []
    if os.path.exists(SUPERSEDED):
        with open(SUPERSEDED) as f:
            old = [json.loads(l) for l in f if l.strip()]
    n = len(recs)
    tin = sum(r["in"] for r in recs)
    tout = sum(r["out"] for r in recs)
    c_in, c_out = tin * RATE_IN, tout * RATE_OUT
    picks = sum(r["choice"] > 0 for r in recs)
    print(f"model {MODEL}   calls {n:,}   picked a candidate {picks:,}   answered none {n - picks:,}")
    print(f"input  tokens {tin:>10,}  (mean {tin / n:,.0f}/call, max {max(r['in'] for r in recs):,})  ${c_in:.4f}")
    print(f"output tokens {tout:>10,}  (mean {tout / n:,.0f}/call, max {max(r['out'] for r in recs):,})  ${c_out:.4f}")
    print(f"total ${c_in + c_out:.4f}   per call ${(c_in + c_out) / n:.6f}   per 1,000 calls ${1000 * (c_in + c_out) / n:.3f}")
    print(f"cost per linked reference ${(c_in + c_out) / max(picks, 1):.6f}")
    if old:
        x = sum(r["in"] for r in old) * RATE_IN + sum(r["out"] for r in old) * RATE_OUT
        print(f"superseded calls {len(old)} (hit max_tokens, re-asked)  ${x:.4f}   billed total ${c_in + c_out + x:.4f}")


if __name__ == "__main__":
    main()
