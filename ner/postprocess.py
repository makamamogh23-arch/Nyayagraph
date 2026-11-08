"""Word-level repair of encoder span predictions (sub-word fragments -> whole-word entities).

The encoder predicts per word-piece; pieces of one word with different tags became separate spans
("Mysore" -> "s", "e", "My", "or"). Repair: expand each span to whole-word boundaries, then merge same-label
spans that touch or overlap; where different labels claim the same word, the label covering more of the
original characters wins. ner/train_encoder.py now decodes at word level directly; this is for predictions
made before that change.

Usage: python -m ner.postprocess GOLD_OR_TEXT_JSONL PRED_IN PRED_OUT
"""
import json
import sys

from ner.common import read_jsonl


def _word_char(c: str) -> bool:
    return c.isalnum()


def repair(text: str, spans: list[dict]) -> list[dict]:
    grown = []
    for s in spans:
        a, b = s["start"], s["end"]
        while a > 0 and _word_char(text[a - 1]) and _word_char(text[a]):
            a -= 1
        while b < len(text) and _word_char(text[b]) and _word_char(text[b - 1]):
            b += 1
        grown.append({"start": a, "end": b, "label": s["label"], "w": s["end"] - s["start"]})
    grown.sort(key=lambda s: (s["start"], -s["end"]))
    out = []
    for s in grown:
        if out and s["start"] <= out[-1]["end"] and not text[out[-1]["end"]:s["start"]].strip():
            prev = out[-1]
            if s["label"] == prev["label"]:
                prev["end"], prev["w"] = max(prev["end"], s["end"]), prev["w"] + s["w"]
                continue
            if s["start"] < prev["end"]:  # different labels overlapping after growth: keep the stronger one
                if s["w"] > prev["w"]:
                    out[-1] = s
                continue
        out.append(s)
    return [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in out]


def main():
    texts = {str(r["id"]): r.get("input", r.get("text")) for r in read_jsonl(sys.argv[1])}
    with open(sys.argv[3], "w") as f:
        for r in read_jsonl(sys.argv[2]):
            f.write(json.dumps({"id": r["id"], "spans": repair(texts[str(r["id"])], r["spans"])}) + "\n")


if __name__ == "__main__":
    main()
