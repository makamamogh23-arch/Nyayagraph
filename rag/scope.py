"""Question scope from the NER entities in the graph: which judgments a question is about, beyond its legal topic.

  court    a named High Court ("appeals from the Allahabad High Court")      -> cases with APPEAL_FROM that court
  lawyer   counsel named with a cue word ("cases argued by Kapil Sibal")     -> cases the lawyer APPEARED_IN
  witness  a PW number together with a named case ("PW-3 in 2025 INSC 529")  -> that case; the witness's paragraphs
                                                                                (WITNESS_IN.paras) are boosted

Detection is deliberately strict (a city next to "High Court", an exact lawyer-name match plus a cue word, a PW
number plus a named case): a wrong filter hides the right judgments, a missed one only leaves the search unscoped.
Several filters intersect. Once the search is restricted, the words that only name the scope are dropped from the
BM25 query (drop_words): inside "appeals from the Bombay High Court" they match nearly every judgment's opening
paragraph (eval/scope_bench.py: gold@10 0.60 with them, 0.93 without). The entities come from the fine-tuned NER
pass (ner/corpus_entities.py), so the prompt says the scope rests on automated extraction.
"""
import re
from dataclasses import dataclass, field

from ner.corpus_entities import HIGH_COURTS, person_key

PW = re.compile(r"\bP\.?\s?W\.?\s?-?\s?(\d{1,3})\b", re.I)
LAWYER_CUE = re.compile(r"\b(?:argued|appear(?:ed|ing|s)?|represent(?:ed|ing|s)?|counsel|advocates?|lawyers?|"
                        r"amicus|instructed)\b", re.I)
NAME_TOKEN = re.compile(r"[A-Z](?:[a-z]+|\.)?")
# words that make a "name" an institution or a role: junk keys like "high court" exist in the NER output
NOT_NAME = {"high", "court", "state", "union", "india", "government", "police", "advocate", "general", "solicitor",
            "counsel", "standing", "senior", "supreme", "bench", "judge", "justice", "section", "act", "code"}


@dataclass
class Scope:
    filters: list[dict] = field(default_factory=list)   # {kind, label, detail, cases}
    case_ids: set[str] | None = None                     # None = whole corpus
    # words that only name the scope ("appeals from the Bombay High Court", "argued by Sidharth Luthra"): once the
    # search is restricted they match nearly every in-scope judgment's opening paragraph, so text search drops them
    drop_words: set[str] = field(default_factory=set)
    paras: dict[str, str] = field(default_factory=dict)  # paragraph pid -> reason, boosted
    notes: list[str] = field(default_factory=list)       # for the prompt

    def restrict(self, cases: set[str]) -> None:
        self.case_ids = set(cases) if self.case_ids is None else self.case_ids & set(cases)


# the place must touch "High Court": "Allahabad High Court", "Madhya Pradesh High Court", "High Court of Delhi",
# "High Court of Judicature at Allahabad"; "the High Court ... Jharkhand" later in the sentence does not count
COURT_BEFORE = re.compile(r"((?:[a-z.&]+\s+){1,3})high\s+court\b")
COURT_AFTER = re.compile(r"\bhigh\s+court\s+((?:of\s+judicature\s+)?(?:of|at|for)\s+(?:the\s+state\s+of\s+)?"
                         r"[a-z.&]+(?:\s+(?:and\s+)?[a-z.&]+){0,2})")


COURT_WORDS = {"appeal", "appeals", "appealed", "from", "high", "court", "courts", "judicature"}
LAWYER_WORDS = {"argued", "appeared", "appearing", "appears", "represented", "representing", "counsel", "advocate",
                "advocates", "lawyer", "lawyers", "senior", "cases", "case", "where", "which"}


def court_phrase_words(q: str) -> set[str]:
    """The words of the "<place> High Court" / "High Court of <place>" phrases in the question."""
    low = " ".join(q.lower().split())
    words = [w for m in COURT_BEFORE.finditer(low) for w in m.group(1).split()]
    words += [w for m in COURT_AFTER.finditer(low) for w in m.group(1).split()]
    return {w.strip(".&") for w in words} | COURT_WORDS


def detect_courts(q: str) -> list[str]:
    """High Courts named right next to "High Court" in the question; a bare "High Court" names none."""
    low = " ".join(q.lower().split())
    out = []
    phrases = [m.group(1).split()[-2:] for m in COURT_BEFORE.finditer(low)]
    phrases += [m.group(1).split() for m in COURT_AFTER.finditer(low)]
    for words in phrases:
        text = " ".join(words)
        for pat, name in HIGH_COURTS:
            if re.search(pat, text):
                if name not in out:
                    out.append(name)
                break
    return out


def detect_pws(q: str) -> list[int]:
    return sorted({int(n) for n in PW.findall(q)})


def detect_lawyers(q: str, keys: set[str]) -> list[str]:
    """Lawyer keys whose full name (2-5 capitalised words / initials) appears in a question that also has a cue
    word. Exact key match only: 'K.K. Venugopal' and 'K K Venugopal' match, 'Venugopal' alone does not."""
    if not LAWYER_CUE.search(q):
        return []
    words = re.findall(r"[A-Za-z]+\.?", q)
    out = []
    for n in (5, 4, 3, 2):
        for i in range(len(words) - n + 1):
            span = words[i:i + n]
            if not all(NAME_TOKEN.fullmatch(w) or (w[0].isupper() and w[1:].islower()) for w in span):
                continue
            key = person_key(" ".join(span))
            # a real name has a word of 3+ letters: keeps "Cr.P.C." from matching a junk key "cr p c"
            if not any(len(w) >= 3 for w in key.split()) or NOT_NAME & set(key.split()):
                continue
            if len(key.split()) >= 2 and key in keys and not any(key in k for k in out):
                out.append(key)
    return out


def neutral_citations(q: str) -> list[str]:
    return [f"{y} INSC {n}" for y, n in re.findall(r"\b(20\d\d)\s*INSC\s*(\d{1,4})\b", q, re.I)]


def analyse(q: str, session, lawyer_keys: set[str], named: list[str]) -> Scope:
    """Scope for a question. `session` is a Neo4j session; `named` are the corpus cases the question names."""
    s = Scope()
    for court in detect_courts(q):
        cases = [r["c"] for r in session.run("MATCH (c:Case)-[:APPEAL_FROM]->(:Court {name: $n}) RETURN c.case_id AS c",
                                             n=court)]
        if cases:
            s.restrict(set(cases))
            s.drop_words |= court_phrase_words(q)
            s.filters.append({"kind": "court", "label": court, "detail": "appeals from", "cases": len(cases)})
            s.notes.append(f"only judgments in appeals from the {court} ({len(cases)} in the corpus)")
    for key in detect_lawyers(q, lawyer_keys):
        rows = list(session.run("""MATCH (l:Lawyer {key: $k})-[e:APPEARED_IN]->(c:Case)
                                   RETURN l.name AS name, c.case_id AS c, e.side AS side, e.senior AS senior""", k=key))
        if rows:
            name = rows[0]["name"]
            sides = {}
            for r in rows:
                sides[r["side"]] = sides.get(r["side"], 0) + 1
            side_text = ", ".join(f"{n} for the {sd}" for sd, n in sorted(sides.items(), key=lambda x: -x[1]))
            s.restrict({r["c"] for r in rows})
            s.drop_words |= set(key.split()) | LAWYER_WORDS
            s.filters.append({"kind": "lawyer", "label": name, "detail": "counsel in", "cases": len(rows)})
            s.notes.append(f"only judgments in which {name} appeared as counsel ({len(rows)} judgments: {side_text}); "
                           "the appearance lists are not among the paragraphs, so state this scope rather than cite it")
    pws = detect_pws(q)
    if pws and named:
        case = named[0]
        rows = list(session.run("""UNWIND $pws AS pw MATCH (w:Witness {wid: $case + '|PW-' + toString(pw)})
                                   -[e:WITNESS_IN]->(:Case) RETURN w.pw AS pw, w.name AS name, e.paras AS paras""",
                                pws=pws, case=case))
        s.restrict({case})
        for r in rows:
            for p in r["paras"]:
                op, no = p.split(":")
                s.paras[f"{case}/{op}/{no}"] = f"PW-{r['pw']}"
            s.filters.append({"kind": "witness", "label": f"PW-{r['pw']} {r['name']}", "detail": case,
                              "cases": 1})
        found = ", ".join(f"PW-{r['pw']} ({r['name']})" for r in rows) or "no numbered witness found"
        s.notes.append(f"only {case}; witness asked about: {found}. Witness numbers and names come from automated "
                       "extraction; rely on how the paragraphs themselves identify the witness")
    return s
