"""InLegalNER -> JSON instruction pairs for QLoRA SFT.

Output per split: data/instruct/{split}.jsonl with
  id, source, instruction, input, output (JSON string), messages (chat format), gold_spans (for eval).
Target JSON: {"entities": [{"label": ..., "text": ...}, ...]} in document order (strings only;
eval/ner_f1.py realigns them to char spans).

Usage: python -m ner.to_instruct [--splits train dev test] [--out data/instruct]
"""
import argparse
import json
import os
from collections import Counter

from ner.common import (JUDGEMENT_LABELS, LABEL_DESC, PREAMBLE_LABELS, is_preamble,
                        load_split, write_jsonl)

SYSTEM = "You are a legal information extraction system for Indian court judgments."


def build_instruction(labels: list[str]) -> str:
    defs = "\n".join(f"- {l}: {LABEL_DESC[l]}" for l in labels)
    return (
        "Extract all named entities from the text below. Allowed labels:\n"
        f"{defs}\n\n"
        "Return only JSON of the form {\"entities\": [{\"label\": LABEL, \"text\": EXACT_SPAN}]}. "
        "Copy each span verbatim from the text, list entities in order of appearance, include repeated "
        "mentions, and return {\"entities\": []} if there are none."
    )


def to_record(doc: dict) -> dict:
    labels = PREAMBLE_LABELS if is_preamble(doc["source"]) else JUDGEMENT_LABELS
    instruction = build_instruction(labels)
    output = json.dumps({"entities": [{"label": s["label"], "text": s["text"]} for s in doc["spans"]]},
                        ensure_ascii=False)
    return {
        "id": doc["id"],
        "source": doc["source"],
        "instruction": instruction,
        "input": doc["text"],
        "output": output,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{instruction}\n\nText:\n{doc['text']}"},
            {"role": "assistant", "content": output},
        ],
        "gold_spans": doc["spans"],
    }


def check(docs: list[dict], split: str):
    n_spans = sum(len(d["spans"]) for d in docs)
    empty = sum(not d["spans"] for d in docs)
    labels = Counter(s["label"] for d in docs for s in d["spans"])
    sources = Counter(d["source"] for d in docs)
    edge_ws = sum(s["text"] != s["text"].strip() for d in docs for s in d["spans"])
    overlaps = sum(a["end"] > b["start"] for d in docs for a, b in zip(d["spans"], d["spans"][1:]))
    lens = sorted(len(d["text"]) for d in docs)
    print(f"[{split}] docs={len(docs)} spans={n_spans} empty_docs={empty} "
          f"edge_whitespace_spans={edge_ws} overlapping_pairs={overlaps} "
          f"chars p50={lens[len(lens)//2]} p99={lens[int(len(lens)*.99)]} max={lens[-1]}")
    print(f"  sources: {dict(sources)}")
    print(f"  labels: {dict(labels.most_common())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--out", default="data/instruct")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for split in args.splits:
        docs = load_split(split)
        check(docs, split)
        write_jsonl(os.path.join(args.out, f"{split}.jsonl"), [to_record(d) for d in docs])


if __name__ == "__main__":
    main()
