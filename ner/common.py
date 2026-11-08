"""InLegalNER loading + string->span alignment shared by ner/ and eval/."""
import json
import re

DATASET = "opennyaiorg/InLegalNER"

LABELS = [
    "COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER", "DATE", "ORG", "GPE",
    "STATUTE", "PROVISION", "PRECEDENT", "CASE_NUMBER", "WITNESS", "OTHER_PERSON",
]
PREAMBLE_LABELS = ["COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER"]
JUDGEMENT_LABELS = [l for l in LABELS if l != "LAWYER"]

LABEL_DESC = {
    "COURT": "name of a court",
    "PETITIONER": "petitioner/appellant/plaintiff party",
    "RESPONDENT": "respondent/defendant party",
    "JUDGE": "name of a judge",
    "LAWYER": "name of an advocate/counsel",
    "DATE": "any date",
    "ORG": "organisation other than courts (police station, bank, company, govt body)",
    "GPE": "geopolitical entity: country, state, city, district, village",
    "STATUTE": "name of an Act or law",
    "PROVISION": "section/article/rule/clause of a statute",
    "PRECEDENT": "cited prior case (party names and/or citation)",
    "CASE_NUMBER": "case/appeal/petition number",
    "WITNESS": "name of a witness",
    "OTHER_PERSON": "any other person's name",
}


def is_preamble(source: str) -> bool:
    return "preamble" in (source or "").lower()


def parse_example(ex: dict) -> dict:
    """HF row -> {id, source, text, spans:[{start,end,label,text}]} (first annotation set, sorted)."""
    text = ex["data"]["text"]
    results = ex["annotations"][0]["result"] if ex.get("annotations") else []
    spans = []
    for r in results or []:
        v = r["value"]
        # ~0.2% of gold spans carry leading/trailing whitespace; trim so offsets match stripped predictions
        s, e = v["start"], v["end"]
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        for lab in v["labels"]:
            spans.append({"start": s, "end": e, "label": lab, "text": text[s:e]})
    spans.sort(key=lambda s: (s["start"], s["end"]))
    return {"id": str(ex["id"]), "source": (ex.get("meta") or {}).get("source", ""), "text": text, "spans": spans}


def load_split(split: str) -> list[dict]:
    from datasets import load_dataset
    return [parse_example(ex) for ex in load_dataset(DATASET, split=split)]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_llm_output(s: str) -> list[dict]:
    """Extract [{label,text}] from raw LLM output; tolerant of prose / code fences / truncation."""
    if isinstance(s, list):
        return s
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, s, re.S)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        ents = obj.get("entities", []) if isinstance(obj, dict) else obj
        return [e for e in ents if isinstance(e, dict) and e.get("text") and e.get("label")]
    # truncated JSON: salvage complete {"label":..,"text":..} objects
    return [{"label": l, "text": t} for l, t in re.findall(r'"label"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"', s)]


def _fuzzy_pattern(s: str) -> str:
    return r"\s+".join(re.escape(tok) for tok in s.split())


def _bounded(s: str) -> str:
    """Exact pattern that refuses partial-word hits ("India" inside "Indian")."""
    pre = r"(?<!\w)" if s[0].isalnum() else ""
    post = r"(?!\w)" if s[-1].isalnum() else ""
    return pre + re.escape(s) + post


def align(text: str, ents: list[dict]) -> list[dict]:
    """Map [{label,text}] (in document order) to char spans. Sequential search from a cursor so repeated
    mentions map to successive occurrences; whole-word hits preferred over substrings; falls back to
    whole-doc, then whitespace/case-insensitive match.
    Unalignable strings are dropped (count as missed recall, not false positives)."""
    out, used, cursor = [], set(), 0
    for e in ents:
        s = e["text"].strip()
        if not s:
            continue
        span = None
        for pat, flags in ((_bounded(s), 0), (re.escape(s), 0), (_fuzzy_pattern(s), re.I)):
            for start_at in (cursor, 0):
                span = next((m.span() for m in re.compile(pat, flags).finditer(text, start_at)
                             if m.span() not in used), None)
                if span:
                    break
            if span:
                break
        if span is None:
            continue
        used.add(span)
        cursor = span[1] if span[0] >= cursor else cursor
        out.append({"start": span[0], "end": span[1], "label": e["label"], "text": text[span[0]:span[1]]})
    return out
