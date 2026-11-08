"""Retrieval smoke test over rag/questions.txt (no LLM calls, free).

Prints, per question: provisions detected, named cases, and the top paragraphs / precedents with why each was picked.
Usage: python -m rag.smoke [--top 6]
"""
import argparse
import time

from rag.retrieve import Retriever


def questions(path: str = "rag/questions.txt") -> list[str]:
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    r = Retriever()
    _ = r.bm25, r.encoder
    print(f"loaded in {time.time() - t0:.0f}s\n")
    for q in questions():
        t = time.time()
        res = r.search(q)
        print(f"Q: {q}   ({time.time() - t:.1f}s)")
        print(f"   provisions: {res['provisions']}   named: {res['named_cases']}")
        for h in res["hits"][:args.top]:
            print(f"   {h.score:.4f} {h.label:24} {h.title[:55]:55} [{', '.join(h.why)}]")
        for h in res["precedents"]:
            print(f"   prec   {h.label:24} {h.title[:55]:55} [cited from {h.cited_from}]")
        print()


if __name__ == "__main__":
    main()
