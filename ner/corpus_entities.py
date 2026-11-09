"""Fine-tuned Qwen NER output over our judgments -> graph tables (data/graph/*.parquet, loaded by graph/load.py).

Inputs: data/ner_corpus/{appearances,witnesses}.jsonl (ner/corpus_inputs.py) and the model's generations
data/ner_corpus/ner_{appearances,witnesses}.jsonl (ner/infer_vllm.py on Kaggle, adapter nyaya-ner-qwen7b-r16).
Every predicted string is aligned back to the source text (ner.common.align); strings that do not occur in the
text are dropped, so nothing the model invents reaches the graph.

  appearances -> Lawyer {key, name}; (Lawyer)-[:APPEARED_IN {side, senior}]->(Case)
                 lower court: (Case)-[:APPEAL_FROM {bench, date, case_numbers, raw}]->(Court {name})
  witnesses   -> Witness {wid, case_id, name, pw}; (Witness)-[:WITNESS_IN {paras}]->(Case)

Usage: python -m ner.corpus_entities
"""
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from ner.common import align, parse_llm_output

D, OUT = "data/ner_corpus", "data/graph"
SOURCE = "qwen2.5-7b-qlora-r16"

# ---------- names --------------------------------------------------------------------------------------------------

HONORIFIC = re.compile(r"^(?:(?:Ms|Mrs|Mr|Dr|Smt|Shri|Sri|Kum|Km)\.?\s+)+", re.I)
DESIGNATION = re.compile(r"[,\s]*(?:\(?\s*(?:Sr\.?|Senior)\s*Adv(?:ocate|s)?\.?\)?|Adv(?:ocate|s)?\.?|A\.?S\.?G\.?|"
                         r"(?:Additional\s+)?Solicitor\s+General|(?:Additional\s+)?Advocate\s+General|"
                         r"A\.?A\.?G\.?|AOR)\s*$", re.I)


def clean_person(s: str) -> str | None:
    s = " ".join(s.split()).strip(" ,.;:-")
    s = HONORIFIC.sub("", s)
    while True:
        t = DESIGNATION.sub("", s).strip(" ,.;:-")
        if t == s:
            break
        s = t
    if not s or re.search(r"\d", s) or len(re.sub(r"[^A-Za-z]", "", s)) < 3:
        return None
    return s


def person_key(s: str) -> str:
    """'K.K. Rai' == 'K. K. Rai' == 'K K RAI'."""
    return " ".join(re.findall(r"[a-z]+", s.lower()))


def display_name(s: str) -> str:
    if s.isupper() or s.islower():
        s = " ".join(w[:1].upper() + w[1:].lower() for w in s.split())
    return re.sub(r"\.(?=[A-Za-z])", ". ", s)  # "K.K. Rai" -> "K. K. Rai"


# ---------- courts -------------------------------------------------------------------------------------------------

# High Courts by any city / state in the court string (principal seat or bench). First match wins.
HIGH_COURTS = [
    (r"punjab|haryana|chandigarh", "High Court of Punjab and Haryana"),
    (r"allahabad|lucknow", "High Court of Judicature at Allahabad"),
    (r"delhi", "High Court of Delhi"),
    (r"madras|mardas|madurai", "High Court of Judicature at Madras"),
    (r"bombay|mumbai|aurangabad|nagpur|goa|panaji", "High Court of Judicature at Bombay"),
    (r"gujarat|ahmedabad", "High Court of Gujarat"),
    (r"karnataka|bangalore|bengaluru|dharwad|gulbarga|kalaburagi", "High Court of Karnataka"),
    (r"kera.a|ernakulam", "High Court of Kerala"),
    (r"patna", "High Court of Judicature at Patna"),
    (r"chhat+is+garh|bilaspur", "High Court of Chhattisgarh"),
    (r"utt?a?rakhand|uttaranchal|nai?nital", "High Court of Uttarakhand"),
    (r"calcutta|kolkata|port blair", "High Court at Calcutta"),
    (r"jharkhand|ranchi", "High Court of Jharkhand"),
    (r"himachal|iimachal|shim\s?la|simla", "High Court of Himachal Pradesh"),
    (r"madhya pradesh|\bm\.?\s?p\b|jabalpur|indore|gwalior", "High Court of Madhya Pradesh"),
    (r"rajasthan|jodhpur|jaipur", "High Court of Judicature for Rajasthan"),
    (r"orissa|odisha|cuttack", "High Court of Orissa"),
    (r"gauhati|guwahati|assam|kohima|itanagar|aizawl", "Gauhati High Court"),
    (r"jammu|kashmir|srinagar|ladakh", "High Court of Jammu and Kashmir"),
    (r"sikkim|gangtok", "High Court of Sikkim"),
    (r"tripura|agartala", "High Court of Tripura"),
    (r"manipur|imphal", "High Court of Manipur"),
    (r"meghala?ya|shill?ong", "High Court of Meghalaya"),
    # undivided Andhra Pradesh sat at Hyderabad until 2019; strings naming both states go there
    (r"hyderabad.*andhra|andhra.*telangana|telangana.*andhra", "High Court of Judicature at Hyderabad"),
    (r"telangana|hyderabad", "High Court for the State of Telangana"),
    (r"andhra|amravati|amaravati", "High Court of Andhra Pradesh"),
]
BENCH = re.compile(r"(?:bench\s*(?:at|,)?\s*|at\s+)([A-Z][a-z]+)\s*(?:bench)?\s*$")


OCR_HIGH_COURT = re.compile(r"\bH\s?i\s?g\.?\s?h\s+C\s?o\s?\S{1,4}?(?=\s+(?:of|at|for|A)\b|\s*$)")
COURTISH = re.compile(r"court|tribunal|judge|magistrate|commission", re.I)


def normalize_court(raw: str) -> tuple[str | None, str | None]:
    """'High Court of Judicature at Bombay, Nagpur Bench, Nagpur' -> ('High Court of Judicature at Bombay', 'Nagpur').
    OCR damage in "High Court" ("High Com1", "High Co mt") is repaired first. None for a bare "High Court" or a
    non-court ("Parliament")."""
    s = " ".join(raw.split()).strip(" ,.")
    s = OCR_HIGH_COURT.sub("High Court", s)
    s = re.sub(r"\bof(?=[A-Z])", "of ", s)
    low = s.lower()
    if not COURTISH.search(s) or low == "high court":
        return None, None
    if "high court" in low:
        for pat, name in HIGH_COURTS:
            if re.search(pat, low):
                seat = None
                m = re.search(r"([A-Z][a-z]+)\s+Bench|Bench(?:\s+at|,)\s+([A-Z][a-z]+)|at\s+([A-Z][a-z]+)\s*$", s)
                if m:
                    seat = next(g for g in m.groups() if g)
                return name, None if seat and seat.lower() in name.lower() else seat
    return s, None


def is_supreme(s: str) -> bool:
    return "supreme" in s.lower()


# ---------- appearances --------------------------------------------------------------------------------------------

# OCR often drops the spaces: "forRespondent", "forthe Appellant"
SIDE_AFTER = re.compile(r"\bfor\s*(?:the\s*)?((?:app|pet|res|rep|int|comp|amicus)[A-Za-z /\-]{0,30})", re.I)
# 2025 layout: the label comes first, "Advs. for the Respondent:\nName, Name"
SIDE_BEFORE = re.compile(SIDE_AFTER.pattern + r"\s*:", re.I)
# the first designation after a name says whether it is senior: "A, B, Sr. Advs." covers both A and B
NEXT_DESIGNATION = re.compile(r"(?P<senior>\b(?:Sr\.?|Senior)\s*A\s?d\s?v)|\bA\s?d\s?v|\bfor\s*(?:the\s*)?[A-Z]|\bIn-?\s?Person", re.I)


def side_of(word: str) -> str:
    w = re.sub(r"[^a-z]", "", word.lower())
    if w.startswith(("app", "pet")) and "appearing" not in w:
        return "appellant"
    if w.startswith(("res", "rep")):
        return "respondent"
    if w.startswith("appearing"):
        return "parties"
    if w.startswith("int"):
        return "intervenor"
    if w.startswith("comp"):
        return "complainant"
    if w.startswith("amicus"):
        return "amicus"
    return "other"


def read(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def predictions(part: str) -> list[tuple[dict, list[dict]]]:
    """(input row, aligned spans) for every generated row of a part."""
    inputs = {r["id"]: r for r in read(f"{D}/{part}.jsonl")}
    out = []
    for g in read(f"{D}/ner_{part}.jsonl"):
        row = inputs.get(g["id"])
        if row:
            out.append((row, align(row["text"], parse_llm_output(g["output"]))))
    return out


def appearances(preds) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    lawyers, appeared, courts, appeal = {}, [], set(), {}
    stats = Counter()
    # side and seniority markers can lie past the end of the piece the model saw (the 2025 headnote layout has no
    # "delivered by" line, so its piece was cut at 1,500 chars): look them up in the full headnote
    heads = pd.read_parquet("data/processed/cases.parquet", columns=["case_id", "headnote"]).set_index("case_id")["headnote"]
    for row, spans in preds:
        text, case_id = row["text"], row["case_id"]
        head, off = heads.get(case_id) or text, row["offset"] if heads.get(case_id) else 0
        law = [s for s in spans if s["label"] == "LAWYER"]
        stats["lawyer_spans"] += len(law)
        for i, s in enumerate(law):
            name = clean_person(s["text"])
            if not name:
                stats["lawyer_dropped"] += 1
                continue
            d = NEXT_DESIGNATION.search(head, off + s["end"])
            senior = bool(d and d.group("senior"))
            before = [b for b in SIDE_BEFORE.finditer(head, 0, off + s["start"])]
            m = before[-1] if before else SIDE_AFTER.search(head, off + s["end"], off + s["end"] + 3000)
            side = side_of(m.group(1)) if m else "other"
            key = person_key(name)
            lawyers.setdefault(key, Counter())[display_name(name)] += 1
            appeared.append((key, case_id, side, senior))
        # court appealed from: first non-Supreme-Court COURT after "From the Judgment" (else the first one at all)
        if case_id in appeal:
            continue
        frm = re.search(r"From\s+the\s+(?:Final\s+)?(?:Judgment|Order)", text, re.I)
        cands = [s for s in spans if s["label"] == "COURT" and not is_supreme(s["text"])]
        after = [s for s in cands if frm and s["start"] >= frm.start()]
        court = (after or cands or [None])[0]
        if not court:
            stats["no_lower_court"] += 1
            continue
        name, seat = normalize_court(court["text"])
        if not name:
            stats["lower_court_unusable"] += 1
            continue
        # the lower court's own case numbers sit between the court and the next "WITH" (connected appeals) or the
        # first lawyer
        first_lawyer = law[0]["start"] if law else len(text)
        w = re.search(r"\n\s*WITH\b", text[court["end"]:])
        if w:
            first_lawyer = min(first_lawyer, court["end"] + w.start())
        date = next((s["text"] for s in spans if s["label"] == "DATE" and frm and frm.start() <= s["start"] < court["start"]), None)
        nums = [" ".join(s["text"].split()) for s in spans
                if s["label"] == "CASE_NUMBER" and court["end"] <= s["start"] < first_lawyer]
        courts.add(name)
        appeal[case_id] = {"case_id": case_id, "court": name, "bench": seat, "date": date,
                           "case_numbers": nums, "raw": " ".join(court["text"].split()), "source": SOURCE}
    lawyer_nodes = pd.DataFrame([(k, c.most_common(1)[0][0]) for k, c in lawyers.items()], columns=["key", "name"])
    edges = pd.DataFrame(appeared, columns=["lawyer", "case_id", "side", "senior"])
    edges = edges.groupby(["lawyer", "case_id"]).agg(
        side=("side", lambda s: Counter(s).most_common(1)[0][0]), senior=("senior", "any")).reset_index()
    edges["source"] = SOURCE
    court_nodes = pd.DataFrame({"name": sorted(courts)})
    return lawyer_nodes, edges, court_nodes, pd.DataFrame(list(appeal.values())), stats


# ---------- witnesses ----------------------------------------------------------------------------------------------

# Which number belongs to a name. Lists come in two styles, and a name inside one has a number on both sides:
#   number first  "PW-2 Ram, PW-3 Shyam"        the number after Ram is followed by a name (Shyam's number)
#   name first    "Ram, PW-2, Shyam, PW-3 and"  the number after Ram is followed by punctuation / "and" / end
# "Ram (PW-3)" and "Ram-PW-3" are unambiguous. An honorific may sit between number and name ("PW-14 Dr. Manju").
PW_TOKEN = r"P\.?\s?W\.?\s?-?\s?(\d+)"
PW_AFTER_TIGHT = re.compile(r"^\s*(?:[(\[]\s*|[-–]\s*)" + PW_TOKEN, re.I)
PW_AFTER = re.compile(r"^\s*[,:]?\s*" + PW_TOKEN + r"(?=\s*(?:[,;.)\]]|and\b|$))", re.I)
PW_BEFORE = re.compile(PW_TOKEN + r"\s*[,:\-–]?\s*\(?\s*(?:(?:Dr|Mr|Mrs|Ms|Smt|Shri|Sri)\.?\s*)?$", re.I)
PW_AFTER_LOOSE = re.compile(r"^\s*[,:]?\s*" + PW_TOKEN, re.I)


def pw_number(text: str, start: int, end: int, span_text: str) -> int | None:
    inside = re.search(PW_TOKEN, span_text, re.I)
    if inside:
        return int(inside.group(1))
    after, before = text[end:end + 30], text[max(0, start - 25):start]
    for m in (PW_AFTER_TIGHT.match(after), PW_AFTER.match(after), PW_BEFORE.search(before),
              PW_AFTER_LOOSE.match(after)):
        if m:
            return int(m.group(1))
    return None


def witnesses(preds) -> tuple[pd.DataFrame, pd.DataFrame, Counter]:
    """One Witness per (case, PW number); a name seen without a number joins the numbered witness of the same name."""
    per_case = defaultdict(lambda: defaultdict(lambda: {"names": Counter(), "paras": set()}))
    unnumbered = defaultdict(list)
    stats = Counter()
    for row, spans in preds:
        text, case_id = row["text"], row["case_id"]
        para = f"{row['opinion']}:{row['para_no']}"
        for s in spans:
            if s["label"] != "WITNESS":
                continue
            stats["witness_spans"] += 1
            name = clean_person(re.sub(r"\(?\s*P\.?\s?W\.?\s?-?\s?\d+\s*\)?", "", s["text"]))
            if not name:
                stats["witness_dropped"] += 1
                continue
            pw = pw_number(text, s["start"], s["end"], s["text"])
            if pw is None:
                unnumbered[case_id].append((name, para))
                continue
            w = per_case[case_id][pw]
            w["names"][display_name(name)] += 1
            w["paras"].add(para)
    nodes, edges = [], []
    for case_id, ws in per_case.items():
        by_key = {person_key(n): pw for pw, w in ws.items() for n in w["names"]}
        for name, para in unnumbered.pop(case_id, []):
            if person_key(name) in by_key:
                ws[by_key[person_key(name)]]["paras"].add(para)
        for pw, w in sorted(ws.items()):
            wid = f"{case_id}|PW-{pw}"
            nodes.append({"wid": wid, "case_id": case_id, "name": w["names"].most_common(1)[0][0], "pw": pw})
            edges.append({"wid": wid, "case_id": case_id, "paras": sorted(w["paras"]), "source": SOURCE})
    stats["unnumbered_cases_skipped"] = len(unnumbered)
    return pd.DataFrame(nodes), pd.DataFrame(edges), stats


def main():
    os.makedirs(OUT, exist_ok=True)
    app = predictions("appearances")
    if app:
        lawyer_nodes, appeared_in, court_nodes, appeal_from, st = appearances(app)
        lawyer_nodes.to_parquet(f"{OUT}/nodes_lawyer.parquet", index=False)
        appeared_in.to_parquet(f"{OUT}/edges_appeared_in.parquet", index=False)
        court_nodes.to_parquet(f"{OUT}/nodes_lower_court.parquet", index=False)
        appeal_from.to_parquet(f"{OUT}/edges_appeal_from.parquet", index=False)
        print(f"appearances: {len(app)} rows | {dict(st)}")
        print(f"  Lawyer {len(lawyer_nodes):,}  APPEARED_IN {len(appeared_in):,} over {appeared_in['case_id'].nunique():,} cases"
              f"  sides {appeared_in['side'].value_counts().to_dict()}  senior {int(appeared_in['senior'].sum()):,}")
        print(f"  lower courts {len(court_nodes)}  APPEAL_FROM {len(appeal_from):,}; top: "
              f"{appeal_from['court'].value_counts().head(5).to_dict()}")
    wit = predictions("witnesses")
    if wit:
        wn, we, st = witnesses(wit)
        wn.to_parquet(f"{OUT}/nodes_witness.parquet", index=False)
        we.to_parquet(f"{OUT}/edges_witness_in.parquet", index=False)
        print(f"witnesses: {len(wit)} rows | {dict(st)} | Witness {len(wn):,} over {wn['case_id'].nunique() if len(wn) else 0:,} cases")


if __name__ == "__main__":
    main()
