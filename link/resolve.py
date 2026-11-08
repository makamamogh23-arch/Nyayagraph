"""Name-based precedent linking on top of the exact citation clusters (graph/build.py).

Three resolvers, all fuzzy first, with Claude Haiku as a tie-breaker for ambiguous cases:
  supra    "Leela Ram (supra)" -> the case this judgment cited in full earlier (candidates = its own CITES)
  corpus   external cluster with a name + year -> in-corpus judgment cited through another reporter
  dedupe   two external clusters with the same name + year -> one case (conflicting OCR'd cites)

Names come from the text just before each citation ("... in Leela Ram v. State of Haryana, (1999) 9 SCC 525")
and from the SCR headnote lists (graph/build.py cluster_names).

LLM decisions are cached in data/cache/link_llm.jsonl (key = hash of the prompt); graph/build.py only reads the
cache, so the graph builds offline and deterministically. Fill the cache with:

  python -m link.resolve --llm          # needs ANTHROPIC_API_KEY (or an `ant auth login` profile)
  python -m link.resolve --eval         # masked-citation accuracy, fuzzy vs fuzzy+LLM
"""
import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from rapidfuzz import fuzz

CACHE = "data/cache/link_llm.jsonl"
MODEL = "claude-haiku-5-5"

# --- names ----------------------------------------------------------------------------------------------------

VS = r"(?:v|vs|versus)"
CONNECT = {"of", "and", "alias", "@", "&", "the", "v", "vs", "versus", "ors", "anr", "or", "for", "by", "through",
           "son", "s/o", "d/o", "w/o", "etc", "dr", "smt", "shri", "sri", "mr", "mrs", "ms", "m/s", "de", "da", "bin"}
LEAD_FILLER = {"of", "and", "the", "&", "@", "in", "see", "also", "vide", "cf", "case", "this",
               "court", "per", "thus", "hence", "further", "therefore", "similarly", "likewise", "judgment", "decision"}
TOKEN = re.compile(r"\(?[\w./@&'’-]+\)?,?")


def _is_name_token(tok: str) -> bool:
    core = tok.strip("(),.").lower()
    if not core:
        return False
    if tok.strip("(),")[:1].isupper() or tok[:1].isdigit():
        return True
    if re.fullmatch(r"[a-z]{1,3}[A-Z]\w*", tok.strip("(),.")):  # OCR glue: "ofHaryana"
        return True
    return core in CONNECT or bool(re.fullmatch(r"[a-z]\.?", core))  # initials typed lowercase by OCR


def name_before(text: str, pos: int, max_chars: int = 160) -> str | None:
    """Case name ending just before pos: walk back over capitalised words and connectors, stop at other words,
    sentence breaks, ';' or an earlier citation's closing bracket/number."""
    win = text[max(0, pos - max_chars):pos]
    win = re.sub(r"(?:reported\s+(?:in|at)|\breported|\bcited\s+as|\bin|\bat)?[\s,:;\[(]*$", "", win, flags=re.I)
    toks = []
    for m in reversed(list(TOKEN.finditer(win))):
        tok = m.group(0)
        if toks and ";" in win[m.end():m.end() + 3]:
            break
        if m.start() == 0 and pos > max_chars:  # word cut by the window edge
            break
        if re.fullmatch(r"\d{1,4}\)?[.,]?", tok) or not _is_name_token(tok):
            break
        toks.append(tok)
    while toks and toks[-1].strip("(),.").lower() in LEAD_FILLER:  # leading "of", "and"
        toks.pop()
    name = " ".join(reversed(toks)).strip(" ,.:;")
    # list markers / preceding clause glued on: "(k) In State of NCT ...", "CrPC. k) In Siddharam ..."
    name = re.sub(r"^.*\b(?:In|See|Also|Vide|Cf)\.?\s+(?!re\b)(?=[A-Z])", "", name)
    name = re.sub(r"^\(?[a-z]{1,4}\)\s*", "", name)
    name = re.sub(r"^(.{4,40}?)\s+\1\b", r"\1", name)  # "Sanjay Dutt Sanjay Dutt v. State"
    if len(re.sub(r"\W", "", name)) < 3 or name.lower() in {"supra", "ibid", "this court", "court"}:
        return None
    return re.sub(rf"\s+{VS}\.?\s+", " v. ", name, flags=re.I)


GENERIC = {"state", "of", "the", "union", "india", "ors", "anr", "and", "others", "another", "through", "its",
           "v", "vs", "versus", "m/s", "ltd", "limited", "public", "prosecutor", "govt", "government", "nct", "delhi"}


def norm(s: str) -> str:
    s = re.sub(r"\([^)]*\)", " ", s.lower())
    return " ".join(re.findall(r"[a-z0-9]+", s))


def parties(name: str) -> list[str]:
    """'Leela Ram v. State of Haryana' -> ['leela ram', 'state haryana'-ish]; distinctive tokens kept per party."""
    out = []
    for p in re.split(rf"\s+{VS}\.?\s+", name, flags=re.I):
        toks = [t for t in norm(p).split() if t not in GENERIC and len(t) > 1]
        if toks:
            out.append(" ".join(toks))
    return out


def name_score(query: str, cand_names: list[str]) -> float:
    """How well a (possibly short, OCR-noisy) mention matches any name of a candidate case. 0-100."""
    q_full, q_parts = norm(query), parties(query)
    best = 0.0
    for n in cand_names:
        best = max(best, fuzz.token_sort_ratio(q_full, norm(n)))
        for qp in q_parts or [q_full]:
            for cp in parties(n):
                if len(qp) >= 4:
                    best = max(best, fuzz.partial_ratio(qp, cp) if len(qp) <= len(cp) else fuzz.ratio(qp, cp))
    return best


# --- LLM tie-break ---------------------------------------------------------------------------------------------

def _cache() -> dict:
    if not os.path.exists(CACHE):
        return {}
    with open(CACHE) as f:
        return {r["key"]: r for r in map(json.loads, f)}


def llm_prompt(kind: str, mention: str, context: str, options: list[str]) -> str:
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options, 1))
    ask = {"supra": f'The judgment refers back to a previously cited case as "{mention} (supra)".',
           "corpus": f'A judgment cites the case "{mention}".',
           }[kind]
    return (f"{ask}\n\nContext:\n{context}\n\nCandidate cases:\n{opts}\n\n"
            "Which candidate is the same case? Names may contain OCR errors (e.g. 'Srilwrm1' for 'Sriharan') and may "
            "use only one party's name. Answer 0 if none of them is clearly the same case.")


def ask_llm(items: list[dict], workers: int = 8) -> dict:
    """items: {key, prompt, n_options} -> {key: choice index (0 = none)}; cached on disk."""
    import anthropic
    from pydantic import BaseModel, Field

    class Pick(BaseModel):
        choice: int = Field(description="1-based index of the matching candidate, 0 if none")

    if not os.environ.get("ANTHROPIC_API_KEY") and os.path.exists(".env"):  # project-local key, never printed
        for line in open(".env"):
            k, _, v = line.strip().removeprefix("export ").partition("=")
            k, v = k.strip(), v.strip().strip("'\"")
            if k == "ANTHROPIC_API_KEY" and v:
                os.environ[k] = v
    client = anthropic.Anthropic()
    cache = _cache()
    todo = [it for it in items if it["key"] not in cache]

    def one(it):
        r = client.messages.parse(
            model=MODEL, max_tokens=4096, output_config={"effort": "low"}, output_format=Pick,
            system="You match Indian court case references to case titles. Be strict: different parties or a "
                   "different dispute means a different case.",
            messages=[{"role": "user", "content": it["prompt"]}])
        c = r.parsed_output.choice if r.parsed_output else 0
        return {"key": it["key"], "choice": c if 0 <= c <= it["n_options"] else 0,
                "in": r.usage.input_tokens, "out": r.usage.output_tokens}

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "a") as f, ThreadPoolExecutor(workers) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(rec) + "\n")
            cache[rec["key"]] = rec
            if i % 50 == 0:
                print(f"  llm {i}/{len(todo)}", flush=True)
    toks = [(cache[k]["in"], cache[k]["out"]) for k in (it["key"] for it in todo)]
    if toks:
        print(f"llm calls {len(toks)}  tokens in {sum(a for a, _ in toks):,} out {sum(b for _, b in toks):,}")
    return {it["key"]: cache[it["key"]]["choice"] for it in items if it["key"] in cache}


def decide(cands: list[tuple[str, float, str]], hi: float, lo: float, margin: float):
    """cands sorted (id, score, display) -> ('fuzzy', id) | ('llm', top-k) | ('none', None)."""
    if not cands or cands[0][1] < lo:
        return "none", None
    second = cands[1][1] if len(cands) > 1 else 0
    if cands[0][1] >= hi and cands[0][1] - second >= margin:
        return "fuzzy", cands[0][0]
    return "llm", cands[:5]


def _key(prompt: str) -> str:
    return hashlib.sha1(f"{MODEL}\n{prompt}".encode()).hexdigest()[:16]


def settle(rows: list[dict], kind: str, use_llm: bool, cache_only: bool = True) -> list[dict]:
    """rows have 'cands' (sorted) + 'mention' + 'context'; fills target/method/score."""
    pending = []
    for r in rows:
        how, val = decide(r.pop("cands"), *THRESH[f"{kind}_global" if r.get("scope") == "global" else kind])
        r["method"], r["target"], r["score"] = how, None, None
        if how == "fuzzy":
            r["target"] = val
        elif how == "llm":
            r["options"] = val
            r["prompt"] = llm_prompt(kind, r["mention"], r["context"], [d for _, _, d in val])
            r["key"] = _key(r["prompt"])
            pending.append(r)
    answers = ask_llm([{"key": r["key"], "prompt": r["prompt"], "n_options": len(r["options"])} for r in pending]) \
        if use_llm and pending else _cache()
    for r in pending:
        a = answers.get(r["key"])
        a = a["choice"] if isinstance(a, dict) else a
        if a is None:
            r["method"] = "unresolved"
        elif a == 0:
            r["method"] = "llm_none"
        else:
            r["method"], r["target"] = "llm", r["options"][a - 1][0]
        for k in ("options", "prompt", "key"):
            r.pop(k, None)
    return rows


THRESH = {"supra": (88, 50, 8), "supra_global": (95, 85, 10), "corpus": (92, 75, 5)}  # (accept, floor, margin)

# --- candidate sets -------------------------------------------------------------------------------------------

def para_texts(paras: pd.DataFrame):
    for r in paras.itertuples():
        yield r, "body", r.text
        for i, t in enumerate(r.footnotes if r.footnotes is not None else []):
            yield r, f"fn{i}", t


def cited_names(paras: pd.DataFrame, cites: pd.DataFrame, key_to: dict) -> pd.DataFrame:
    """One row per citation group with the name printed before it: case_id, dst, name, para, pos."""
    text = {(r.case_id, r.opinion, r.para_no, src): t for r, src, t in para_texts(paras)}
    first = cites.sort_values("start").drop_duplicates("group")
    rows = []
    for c in first.itertuples():
        src = c.group.split("/")[3]
        t = text.get((c.case_id, c.opinion, c.para_no, src))
        dst = key_to.get(c.key)
        if t is None or dst is None or dst == c.case_id:
            continue
        rows.append({"case_id": c.case_id, "dst": dst, "name": name_before(t, c.start),
                     "para": f"{c.opinion}:{c.para_no}", "order": (c.opinion, c.para_no, src, c.start)})
    return pd.DataFrame(rows)


def node_names(named: pd.DataFrame, titles: dict) -> dict:
    """node id -> list of known names (headnote title + every name printed before its cites)."""
    out = defaultdict(set)
    for n, t in titles.items():
        if isinstance(t, str) and t:
            out[n].add(t)
    for r in named.dropna(subset=["name"]).itertuples():
        if re.search(rf"\s{VS}\.?\s", r.name, re.I):  # only full "X v. Y" names: short ones are ambiguous
            out[r.dst].add(r.name)
    return {k: sorted(v) for k, v in out.items()}


SUPRA = re.compile(r"\(\s*supra\s*\)", re.I)


def _score_all(mention: str, ids, names: dict) -> list[tuple[str, float, str]]:
    # tie-break on id: `ids` may be a set, whose order changes per process (hash seed), and the prompt
    # text (hence the cache key) must not
    return sorted(((d, name_score(mention, names.get(d, [])), "; ".join(names.get(d, [])[:2]) or d) for d in ids),
                  key=lambda x: (-x[1], x[0]))


def token_index(names: dict) -> dict:
    """distinctive party token -> node ids (for '(supra)' whose full citation the parser missed)."""
    idx = defaultdict(set)
    for node, ns in names.items():
        for n in ns:
            for p in parties(n):
                for t in p.split():
                    if len(t) >= 4:
                        idx[t].add(node)
    return idx


def supra_mentions(paras: pd.DataFrame, named: pd.DataFrame, names: dict) -> list[dict]:
    """Candidates = cases this judgment cites. If none fits (its full citation was missed or never printed),
    fall back to every named case sharing a rare party token, under the stricter 'global' thresholds."""
    per_case = named.groupby("case_id")["dst"].agg(lambda s: sorted(set(s))).to_dict()
    idx = token_index(names)
    rows = []
    for r, src, t in para_texts(paras):
        for m in SUPRA.finditer(t or ""):
            mention = name_before(t, m.start())
            if not mention:
                continue
            scored, scope = _score_all(mention, per_case.get(r.case_id, []), names), "judgment"
            if not scored or scored[0][1] < THRESH["supra"][1]:
                toks = {x for p in parties(mention) for x in p.split() if len(x) >= 4}
                pool = set().union(*(idx[x] for x in toks if 0 < len(idx.get(x, ())) <= 50)) if toks else set()
                scored, scope = _score_all(mention, pool, names), "global"
                if len(toks) < 2 and scored:  # one word ("Airways") is too thin to accept without the LLM
                    scored = [(d, min(sc, THRESH["supra_global"][0] - 1), disp) for d, sc, disp in scored]
            rows.append({"case_id": r.case_id, "para": f"{r.opinion}:{r.para_no}", "src": src, "mention": mention,
                         "scope": scope, "context": t[max(0, m.start() - 300):m.end() + 100], "cands": scored})
    return rows


def corpus_matches(nodes: pd.DataFrame, names: dict) -> list[dict]:
    """External clusters (year within corpus range) -> in-corpus case of the same year by party names."""
    inc = nodes[nodes["in_corpus"]]
    by_year = {y: g for y, g in inc.groupby("year")}
    rows = []
    for r in nodes[~nodes["in_corpus"] & (nodes["year"] >= inc["year"].min())].itertuples():
        ns = [n for n in names.get(r.case_id, []) if re.search(rf"\s{VS}\.?\s", n, re.I)]
        g = by_year.get(r.year)
        if not ns or g is None:
            continue
        scored = []
        for c in g.itertuples():
            title = f"{c.petitioner} v. {c.respondent}"
            s = max(min(fuzz.token_set_ratio(norm(n), norm(title)), _parties_both(n, title)) for n in ns)
            scored.append((c.case_id, s, f"{title} ({c.citation}, decided {c.decision_date})"))
        scored.sort(key=lambda x: -x[1])
        rows.append({"case_id": r.case_id, "mention": ns[0], "context": f"Cited as: {'; '.join(ns[:3])}; year {r.year}",
                     "cands": scored})
    return rows


def _parties_both(a: str, b: str) -> float:
    """Both sides must match (order-free): min over a's parties of best match in b's parties."""
    pa, pb = parties(a), parties(b)
    if len(pa) < 2 or not pb:  # a truncated name ("& Ors. v. State of U.P.") can match anything
        return 0.0
    return min(max(fuzz.token_set_ratio(x, y) for y in pb) for x in pa)


def dedupe_external(nodes: pd.DataFrame, names: dict, hi: float = 95) -> dict:
    """ext id -> canonical ext id for same-year clusters whose full names match (no conflicting SCC cite)."""
    ext = nodes[~nodes["in_corpus"]]
    merge = {}
    for y, g in ext.groupby("year"):
        items = [(r.case_id, [norm(n) for n in names.get(r.case_id, []) if re.search(rf"\s{VS}\.?\s", n, re.I)],
                  {k for k in r.keys if k.startswith("SCC|")}) for r in g.itertuples()]
        items = [it for it in items if it[1]]
        for i, (a, na, ka) in enumerate(items):
            for b, nb, kb in items[i + 1:]:
                if ka and kb and not ka & kb:
                    continue  # two different SCC cites = two cases
                if max(fuzz.token_sort_ratio(x, z) for x in na for z in nb) >= hi:
                    root = merge.get(a, a)
                    merge[b] = merge.get(b, root) if b in merge else root
    return merge


# --- entry points ----------------------------------------------------------------------------------------------

def resolve_all(paras, cites, key_to, nodes, titles, use_llm=False) -> dict:
    named = cited_names(paras, cites, key_to)
    names = node_names(named, titles)
    supra = settle(supra_mentions(paras, named, names), "supra", use_llm)
    corpus = settle(corpus_matches(nodes, names), "corpus", use_llm)
    return {"named": named, "names": names, "supra": pd.DataFrame(supra), "corpus": pd.DataFrame(corpus),
            "dedupe": dedupe_external(nodes, names)}


def masked_eval(paras, cites, key_to, titles, nodes, use_llm, n=600, seed=0) -> None:
    """Hide each citation, keep only the first party of the name printed before it (how '(supra)' refers back),
    and resolve it among everything the same judgment cites. Gold = the citation's own cluster."""
    named = cited_names(paras, cites, key_to).dropna(subset=["name"])
    names = node_names(named, titles)
    per_case = named.groupby("case_id")["dst"].agg(lambda s: sorted(set(s))).to_dict()
    sample = named[named["name"].str.contains(rf"\s{VS}\.?\s", case=False, regex=True)]
    sample = sample[sample["case_id"].map(lambda c: len(per_case.get(c, [])) >= 2)].sample(n, random_state=seed)
    rows, gold = [], []
    for r in sample.itertuples():
        first = re.split(rf"\s+{VS}\.?\s+", r.name, flags=re.I)[0]
        scored = sorted(((d, name_score(first, names.get(d, [])), "; ".join(names.get(d, [])[:2]) or d)
                         for d in per_case[r.case_id]), key=lambda x: -x[1])
        rows.append({"case_id": r.case_id, "mention": first, "context": f"... {first} (supra) ...", "cands": scored})
        gold.append(r.dst)
    out = settle(rows, "supra", use_llm)
    df = pd.DataFrame(out).assign(gold=gold)
    merge = dedupe_external(nodes, names)  # same case under two reporter cites counts once, as in the graph
    canon = lambda x: merge.get(x, x)
    df["correct_raw"] = df["target"] == df["gold"]
    df["correct"] = df["target"].map(canon) == df["gold"].map(canon)
    print(f"masked eval n={len(df)}  candidates/judgment median {pd.Series([len(per_case[c]) for c in df.case_id]).median():.0f}")
    print(df.groupby("method").agg(n=("correct", "size"), correct=("correct", "sum"),
                                   correct_raw=("correct_raw", "sum")).to_string())
    linked = df[df["target"].notna()]
    print(f"precision {linked['correct'].mean():.3f}  recall {df['correct'].mean():.3f}  "
          f"(raw ids: {linked['correct_raw'].mean():.3f} / {df['correct_raw'].mean():.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="call Claude Haiku for ambiguous cases (cached)")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    from graph.build import load_for_resolve
    paras, cites, key_to, nodes, titles = load_for_resolve()
    if args.eval:
        masked_eval(paras, cites, key_to, titles, nodes, args.llm)
        return
    res = resolve_all(paras, cites, key_to, nodes, titles, use_llm=args.llm)
    s, c = res["supra"], res["corpus"]
    print(f"supra mentions {len(s):,}:", s["method"].value_counts().to_dict())
    print(f"external->corpus candidates {len(c):,}:", c["method"].value_counts().to_dict())
    print(f"external duplicates merged: {len(res['dedupe']):,}")


if __name__ == "__main__":
    main()
