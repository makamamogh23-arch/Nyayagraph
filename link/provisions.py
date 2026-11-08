"""Statutory provision parser: "Sections 302, 307 r/w 34 IPC" -> [(IPC,302),(IPC,307),(IPC,34)].

Act resolution order (act_source):
  explicit  Act named right after the provision list ("... of the NDPS Act", "... IPC")
  alias     anaphora ("of the Act", "the Code") resolved via the case's own definitions
            ("Code of Criminal Procedure, 1973 (for short 'the Code')")
  default   undefined "the Code" -> CrPC; default_case / default_ipc when the judgment's own explicit cites of
            that section, or an IPC-only section number (498A), say otherwise
  article   bare "Article N" -> Constitution
  case      bare section -> Act this case explicitly uses for the same section number
  inferred  bare section -> nearest Act named earlier in the same paragraph
  none      unresolved (act = None)

Writes data/processed/provisions.parquet:
  case_id, opinion, para_no, act, section, subsection, act_source, raw, start, end

Usage: python -m link.provisions
"""
import re
from collections import Counter

import pandas as pd

from link.acts import ALIAS_RE, GENERIC_RE, merge_variants, normalize, strip_prefix

KW = r"(?:Sections?|Secs?\.|Ss?\.|u/ss?\.?|U/[Ss]s?\.?|Articles?|Arts?\.|Rules?|Clauses?)"
# letter suffix only when attached or hyphenated ("498A", "304-B"), never "34 IPC"
NUM = r"\d{1,4}(?:(?:\s?-\s?)?[A-Z]{1,2}(?![A-Za-z]))?(?:\s?\(\s?[\w]{1,5}\s?\))*"
BARE_SUB = r"\(\s?\w{1,4}\s?\)"  # "13(1)(d) and (2)" -> (2) is a sub-section of the previous section
CONN = r"(?:\s*(?:,|/|&|and|or|to|r/w|r\.w\.|read\s+with|alongwith|along\s+with|with)\s*(?:" + KW + r"\s*)?)"
ACT_TAIL = (r"(?P<tail>\s*,?\s*(?:of\s+(?:the\s+)?|under\s+(?:the\s+)?|in\s+(?:the\s+)?)?"
            r"(?P<act>(?:the\s+)?(?:said|same|this|that)?\s*(?:Act|Code|Rules)\b|" + ALIAS_RE + "|" + GENERIC_RE + r"))?")
# between the numbers and "of the Act": a spaced suffix of the last section ("498 A of the Code" = 498A),
# or a Part / Chapter qualifier ("304 Part II of the IPC", "149 to 152 of Chapter XI of the Code")
QUAL = (r"(?P<qual>\s*,?\s*(?:\(?Part[- ]?[IVX]{1,3}\)?|Chapter\s+[IVXL]{1,6}|"
        r"\(?(?P<suf>[A-Z]{1,2})\)?)(?=\s*,?\s*of\s))?")
PROV = re.compile(rf"(?P<kw>{KW})\s*(?P<nums>{NUM}(?:{CONN}(?:{NUM}|{BARE_SUB}))*){QUAL}{ACT_TAIL}")
RANGE = re.compile(rf"({NUM})\s*to\s*({NUM})")

# case-insensitive only in the keyword part: GENERIC_RE relies on capitalised words
DEF = re.compile(
    r"(?P<name>" + ALIAS_RE + "|" + GENERIC_RE + r")\s*,?\s*(?:\d{4})?\s*\(\s*(?i:hereinafter|for\s+short|in\s+short|"
    r"referred\s+to\s+as|for\s+brevity)[^)‘'\"“]{0,40}[‘'\"“]\s*(?i:the\s+)?(?P<short>[^’'\"”]{1,40})[’'\"”]\s*\)")
DEFAULTS = {"code": "Code of Criminal Procedure, 1973"}  # criminal appeals: bare "the Code" is CrPC unless defined
CODES = {"Indian Penal Code, 1860", "Code of Criminal Procedure, 1973", "Code of Civil Procedure, 1908"}
# lettered sections only the IPC has (from explicit IPC cites in the corpus, minus CrPC's lettered sections);
# plain numbers are not used: 485-565 are also old (1898) CrPC sections, e.g. 561A
IPC_ONLY = {"120A", "120B", "121A", "124A", "153A", "153B", "165A", "171F", "174A", "195A", "228A", "229A", "304A",
            "304B", "326A", "326B", "354A", "354B", "354C", "354D", "363A", "364A", "366A", "370A", "376A", "376AB",
            "376B", "376C", "376D", "376DA", "376DB", "376E", "477A", "489B", "489C", "498A", "506B"}


def case_aliases(texts: list[str]) -> dict[str, str]:
    votes: dict[str, Counter] = {}
    for t in texts:
        for m in DEF.finditer(t):
            canon = normalize(m["name"])
            if canon:
                votes.setdefault(m["short"].strip().lower(), Counter())[canon] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items()}


def split_nums(nums: str) -> list[tuple[str, str]]:
    """'302, 307 r/w 34' -> [('302',''),('307',''),('34','')]; '13(1)(d)' -> ('13','(1)(d)');
    small numeric ranges '3 to 5' expanded."""
    out = []
    for a, b in RANGE.findall(nums):
        if a.isdigit() and b.isdigit() and 0 < int(b) - int(a) <= 10:
            out += [(str(n), "") for n in range(int(a) + 1, int(b))]
    for tok in re.findall(rf"{NUM}|{BARE_SUB}", nums):
        if tok.startswith("("):
            if out:
                out.append((out[-1][0], re.sub(r"\s", "", tok)))
            continue
        m = re.match(r"(\d{1,4})(?:\s?-\s?)?([A-Z]{0,2})((?:\s?\(\s?\w{1,5}\s?\))*)", tok)
        out.append((m[1] + m[2], re.sub(r"\s", "", m[3])))
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def resolve_act(surface: str | None, aliases: dict) -> tuple[str | None, str]:
    if not surface:
        return None, "none"
    # "Chapter XI of the Code", "Part II of the Indian Penal Code", "A of the Code": the Act is what follows
    surface = strip_prefix(surface)
    canon = normalize(surface)
    if canon:
        return canon, "explicit"
    key = re.sub(r"^(?:the|said|same|this|that)\s+", "", surface.strip(), flags=re.I).lower()
    key = re.sub(r"^(?:said|same|this|that)\s+", "", key)
    if key in aliases:
        return aliases[key], "alias"
    if "the " + key in aliases:
        return aliases["the " + key], "alias"
    if key in DEFAULTS:
        return DEFAULTS[key], "default"
    return None, "none"


def extract(text: str, aliases: dict) -> list[dict]:
    rows, last_act = [], None
    acts_seen = [(m.start(), normalize(m.group(0))) for m in re.finditer(ALIAS_RE + "|" + GENERIC_RE, text)]
    for m in PROV.finditer(text):
        act, src = resolve_act(m["act"], aliases)
        if act is None and m["kw"].lower().startswith("art"):
            act, src = "Constitution of India", "article"
        if act is None:
            prior = [a for s, a in acts_seen if s < m.start() and a]
            if prior:
                act, src = prior[-1], "inferred"
        secs = split_nums(m["nums"])
        if m["suf"] and secs and secs[-1][0].isdigit():
            secs[-1] = (secs[-1][0] + m["suf"], secs[-1][1])
        for sec, sub in secs:
            rows.append({"act": act, "section": sec, "subsection": sub, "act_source": src,
                         "raw": m.group(0).strip(), "start": m.start(), "end": m.end()})
        last_act = act or last_act
    return rows


def fix_ipc_ocr(df: pd.DataFrame) -> pd.DataFrame:
    """IPC ends at s.511: OCR "1208"/"3048" = 120B/304B; "304I"/"304II" = s.304 Part I/II.
    Applied to any row whose explicit Act is IPC or that has no Act yet (bare sections inherit an Act later)."""
    maybe_ipc = df["act"].isna() | (df["act"] == "Indian Penal Code, 1860")
    num = df["section"].str.extract(r"^(\d+)")[0].astype(float)
    ocr_b = maybe_ipc & df["section"].str.fullmatch(r"\d{3,4}8") & (num > 511) & (num // 10 <= 511)
    df.loc[ocr_b, "section"] = df.loc[ocr_b, "section"].str[:-1] + "B"
    df.loc[maybe_ipc & df["section"].str.fullmatch(r"304I{1,2}"), "section"] = "304"
    ocr_one = df["section"].str.fullmatch(r"\d{1,3}I") & (df["section"] != "304I")  # "Section 40 I" = 401
    df.loc[ocr_one, "section"] = df.loc[ocr_one, "section"].str[:-1] + "1"
    return df


def main():
    paras = pd.read_parquet("data/processed/paragraphs.parquet")
    rows = []
    for case_id, g in paras.groupby("case_id", sort=False):
        aliases = case_aliases(g["text"].tolist())
        for r in g.itertuples():
            for h in extract(r.text, aliases):
                h.update(case_id=case_id, opinion=r.opinion, para_no=r.para_no)
                rows.append(h)
    df = fix_ipc_ocr(pd.DataFrame(rows))
    merged = merge_variants(df["act"].value_counts().to_dict())
    df["act"] = df["act"].map(lambda a: merged.get(a, a))
    print(f"act spellings merged: {len(merged)} names, {df['act'].isin(set(merged.values())).sum():,} rows on merged names")

    # case-level: a bare "Section 302" takes the Act this case explicitly pairs with 302 (most common)
    sure = df[df["act_source"].isin(["explicit", "alias"])]
    case_map = sure.groupby(["case_id", "section"])["act"].agg(lambda s: s.value_counts().index[0])
    keys = list(zip(df["case_id"], df["section"]))
    case_act = pd.Series([case_map.get(k) for k in keys], index=df.index)
    inferred = df["act_source"] == "inferred"
    agree = (df.loc[inferred, "act"] == case_act[inferred])[case_act[inferred].notna()]
    print(f"paragraph-inferred Acts agreeing with case-level evidence: {agree.mean():.3f} (n={len(agree)})")
    fill = df["act_source"].isin(["inferred", "none"]) & case_act.notna()
    df.loc[fill, "act"], df.loc[fill, "act_source"] = case_act[fill], "case"

    # bare "the Code" with no definition in the judgment: CrPC by default, unless this judgment explicitly ties
    # the section number to another Code, or the section exists only in the IPC
    dflt = df["act_source"] == "default"
    by_case = dflt & case_act.isin(CODES) & (case_act != df["act"])
    df.loc[by_case, "act"], df.loc[by_case, "act_source"] = case_act[by_case], "default_case"
    by_sec = (df["act_source"] == "default") & df["section"].isin(IPC_ONLY)
    df.loc[by_sec, "act"], df.loc[by_sec, "act_source"] = "Indian Penal Code, 1860", "default_ipc"
    # the same holds whatever the text says ("498A of the CrPC", a 'Code' defined as CrPC then cited for 304B)
    crpc_ipc = (df["act"] == "Code of Criminal Procedure, 1973") & df["section"].isin(IPC_ONLY)
    df.loc[crpc_ipc, "act"] = "Indian Penal Code, 1860"
    print(f"bare 'the Code': {int(dflt.sum()):,} rows; re-assigned by case evidence {int(by_case.sum())}, "
          f"by IPC-only section {int(by_sec.sum())}; other CrPC rows with an IPC-only section {int(crpc_ipc.sum())}")

    # second, weaker pass: corrected 'the Code' rows (95% agree with explicit cites where both exist) as evidence
    # for bare sections the explicit cites left unresolved
    weak = df[df["act_source"].isin(["explicit", "alias", "default", "default_case", "default_ipc"])]
    weak_map = weak.groupby(["case_id", "section"])["act"].agg(lambda s: s.value_counts().index[0])
    left = df["act_source"].isin(["inferred", "none"])
    weak_act = pd.Series([weak_map.get(k) if l else None for k, l in zip(zip(df["case_id"], df["section"]), left)],
                         index=df.index)
    fill2 = left & weak_act.notna()
    df.loc[fill2, "act"], df.loc[fill2, "act_source"] = weak_act[fill2], "case"

    df.to_parquet("data/processed/provisions.parquet", index=False)

    print(f"provision mentions={len(df)}  paragraphs with >=1: "
          f"{df[['case_id','opinion','para_no']].drop_duplicates().shape[0] / len(paras):.3f}  "
          f"cases with >=1: {df['case_id'].nunique() / paras['case_id'].nunique():.3f}")
    print("act_source:", df["act_source"].value_counts(normalize=True).round(3).to_dict())
    print(f"act resolved (excl. inferred): {df['act_source'].isin(['explicit','alias','article','case']).mean():.3f}  "
          f"(incl. inferred): {df['act'].notna().mean():.3f}")
    print("top acts:", df["act"].value_counts().head(12).to_dict())
    top = df[df["act"] == "Indian Penal Code, 1860"]["section"].value_counts().head(12).to_dict()
    print("top IPC sections:", top)


if __name__ == "__main__":
    main()
