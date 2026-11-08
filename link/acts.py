"""Act name normalization: surface form / abbreviation -> canonical Act name."""
import re

# (canonical, pattern over the surface form). Order matters: specific before generic.
ACTS = [
    ("Indian Penal Code, 1860", r"I\.?\s?P\.?\s?C\.?|Indian\s+Penal\s+Code|Penal\s+Code"),
    ("Code of Criminal Procedure, 1973", r"Cr\.?\s?P\.?\s?C\.?|Crl\.?\s?P\.?\s?C\.?|Code\s+of\s+Criminal\s+Procedure|Criminal\s+Procedure\s+Code"),
    ("Code of Civil Procedure, 1908", r"C\.?\s?P\.?\s?C\.?|Code\s+of\s+Civil\s+Procedure|Civil\s+Procedure\s+Code"),
    ("Indian Evidence Act, 1872", r"(?:Indian\s+)?Evidence\s+Act"),
    ("Constitution of India", r"Constitution(?:\s+of\s+India)?"),
    # BNSS before BNS: alternation is ordered, and 'BNS' would match the start of 'BNSS'
    ("Bharatiya Nagarik Suraksha Sanhita, 2023", r"B\.?\s?N\.?\s?S\.?\s?S\.?|Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita"),
    ("Bharatiya Nyaya Sanhita, 2023", r"B\.?\s?N\.?\s?S\.?|Bharatiya\s+Nyaya\s+Sanhita"),
    ("Bharatiya Sakshya Adhiniyam, 2023", r"B\.?\s?S\.?\s?A\.?|Bharatiya\s+Sakshya\s+Adhiniyam"),
    ("Narcotic Drugs and Psychotropic Substances Act, 1985", r"N\.?\s?D\.?\s?P\.?\s?S\.?(?:\s+Act)?|Narcotic\s+Drugs\s+and\s+Psychotropic\s+Substances\s+Act"),
    ("Prevention of Corruption Act, 1988", r"P\.?\s?C\.?\s+Act|Prevention\s+of\s+Corruption\s+Act"),
    ("Negotiable Instruments Act, 1881", r"N\.?\s?I\.?\s+Act|Negotiable\s+Instruments?\s+Act"),
    ("Arms Act, 1959", r"Arms\s+Act"),
    ("Protection of Children from Sexual Offences Act, 2012", r"POCSO(?:\s+Act)?|Protection\s+of\s+Children\s+from\s+Sexual\s+Offences\s+Act"),
    ("Scheduled Castes and the Scheduled Tribes (Prevention of Atrocities) Act, 1989",
     r"S\.?\s?C\.?\s*(?:/|,|-|and|&)\s*S\.?\s?T\.?\s*(?:\(\s*P\.?\s?o?\.?\s?A\.?\s*\)|\(?Prevention\s+of\s+Atrocities\)?)?\s+Act|"
     r"(?:Prevention\s+of\s+)?Atrocities\s+Act|"
     r"Scheduled\s+Castes?\s*(?:and|&|/)\s*(?:the\s+)?Scheduled\s+Tribes?,?\s*(?:\(?Prevention\s+(?:of|and)\s+Atrocities\)?)?,?\s+Act"),
    ("Juvenile Justice (Care and Protection of Children) Act", r"J\.?\s?J\.?\s+Act|Juvenile\s+Justice(?:\s*\(Care\s+and\s+Protection\s+of\s+Children\))?\s+Act"),
    ("Prevention of Money-Laundering Act, 2002", r"PMLA|Prevention\s+of\s+Money[\s-]Laundering\s+Act|Money\s+Laundering\s+Act"),
    ("Dowry Prohibition Act, 1961", r"Dowry\s+Prohibition\s+Act|D\.?\s?P\.?\s+Act"),
    ("Unlawful Activities (Prevention) Act, 1967", r"UAPA|U\.?\s?A\.?\s?P\.?\s?A\.?|Unlawful\s+Activities\s*\(Prevention\)\s+Act"),
    ("Terrorist and Disruptive Activities (Prevention) Act, 1987",
     r"T\.?A\.?\s?D\.?\s?A\.?(?:\s+Act)?|Terrorists?\s+and\s+Disruptive(?:\s+Activities)?(?:\s*\(Prevention\))?\s+Act"),
    ("Uttar Pradesh Gangsters and Anti-Social Activities (Prevention) Act, 1986",
     r"(?:U\.?\s?P\.?\s*|Uttar\s+Pradesh\s+)?Gangsters?(?:\s+(?:and|&)\s+Anti[-\s]?Social(?:\s+Activities)?"
     r"(?:\s*\(Prevention\))?)?\s+Act"),
    ("Maharashtra Control of Organised Crime Act, 1999", r"MCOCA|Maharashtra\s+Control\s+of\s+Organi[sz]ed\s+Crime\s+Act"),
    ("Protection of Women from Domestic Violence Act, 2005", r"D\.?\s?V\.?\s+Act|Domestic\s+Violence\s+Act|Protection\s+of\s+Women\s+from\s+Domestic\s+Violence\s+Act"),
    ("Information Technology Act, 2000", r"I\.?\s?T\.?\s+Act|Information\s+Technology\s+Act"),
    ("Explosive Substances Act, 1908", r"Explosive\s+Substances\s+Act"),
    ("Prevention of Food Adulteration Act, 1954", r"(?:Prevention\s+of\s+)?Food\s+Adulteration\s+Act|P\.?\s?F\.?\s?A\.?(?:\s+Act)?"),
    ("Drugs and Cosmetics Act, 1940", r"Drugs\s+and\s+Cosmetics\s+Act"),
    ("Customs Act, 1962", r"Customs\s+Act"),
    ("Income Tax Act, 1961", r"Income[\s-]Tax\s+Act"),
    ("Companies Act", r"Companies\s+Act"),
    ("Delhi Special Police Establishment Act, 1946", r"Delhi\s+Special\s+Police\s+Establishment\s+Act|D\.?\s?S\.?\s?P\.?\s?E\.?\s+Act"),
    ("General Clauses Act, 1897", r"General\s+Clauses\s+Act"),
    ("Probation of Offenders Act, 1958", r"Probation\s+of\s+Offenders\s+Act"),
]
_COMPILED = [(c, re.compile(rf"(?:the\s+)?(?:{p})(?:\s*,?\s*\d{{4}})?", re.I)) for c, p in ACTS]
# capitalised words with lowercase connectors: "Contempt of Courts Act", "Juvenile Justice (Care and Protection ...) Rules"
_WORDS = r"(?:(?:[A-Z]|\((?=[A-Z]))[\w()&'.,/-]*\s+(?:(?:of|and|the|for|in|on|to|from|&)\s+)*){1,12}?"
_KINDS = r"(?:Act|Code|Sanhita|Adhiniyam|Rules|Regulations)"
_GENERIC = re.compile(rf"(?:the\s+)?({_WORDS}{_KINDS})(?:\s*,?\s*(\d{{4}}))?$")

# regex fragment for spotting an Act name right after a provision list
ALIAS_RE = "|".join(f"(?:{p})" for _, p in ACTS)
GENERIC_RE = rf"{_WORDS}{_KINDS}(?:\s*,?\s*\d{{4}})?"

CASE_SENSITIVE_ABBR = re.compile(r"^(?:[A-Z]\.?\s?){2,6}$")


# Leading material the generic pattern swallows: a section suffix or sub-part ("498 A of the Code",
# "(C) of the NDPS Act", "13 A(1) of the PC Act"), a Part / Chapter / Schedule / Order ("Part II of the IPC"),
# a stray OCR letter ("E Indian Penal Code", "Part-I of G the Code"), or sentence debris ("In the", "Hence, the").
_PREFIX = re.compile(
    r"^(?:\(?(?:[A-Z]{1,2}|[A-Z]\s[A-Z])\)?(?:\(\s?\w{1,3}\s?\))?\)?\s*,?\s*(?:of|in|to|under)\s+"
    r"|[A-Z]ofthe\s*"
    r"|\(?(?:Part|Chapter|Schedule|Order)[- ]?\(?[IVXL]{1,6}A?\)?\)?(?:\s*,?\s*(?:of|to)\s+|,\s+(?=(?:the\s+)?[A-Z]))"
    r"|(?:P\.C\.,\s*)?Sec\.\s?[\w()]+\s+"
    r"|(?:Sections?|Rules?|Articles?|Clauses?)\s+(?:of|under)\s+"
    r"|[A-Z]\s+(?=(?:the\s+)?[A-Z][a-z]{2})"
    r"|(?:In|Hence,|Act,)\s+(?=the\s)"
    r"|the\s+)")


def strip_prefix(name: str) -> str:
    s = re.sub(r"\s+", " ", name).strip(" ,.;:")
    while True:
        t = _PREFIX.sub("", s, count=1).strip(" ,.;:")
        if t == s:
            return s
        s = t


def normalize(name: str) -> str | None:
    """Surface form -> canonical name; unknown full Act names are title-cased with year kept;
    anaphora ('the Act', 'the Code', 'this Act') -> None (resolve via case aliases)."""
    s = strip_prefix(name)
    if re.fullmatch(r"(?:(?:the|this|said|that|same)\s+)*(?:Act|Code|Rules|Regulations|Sanhita)", s, re.I) or not s:
        return None
    for canon, pat in _COMPILED:
        if pat.fullmatch(s):
            # BNSS / BNS / CPC / CrPC abbreviations overlap: require exact abbreviation case
            if CASE_SENSITIVE_ABBR.match(s.replace(",", "")) and s.upper() != s:
                continue
            return canon
    m = _GENERIC.fullmatch(s)
    if m:
        base = re.sub(r"^(?:the|this)\s+", "", m.group(1), flags=re.I)
        return f"{base}, {m.group(2)}" if m.group(2) else base
    return None


# --- spelling / OCR variant merge ------------------------------------------------------------------------------
# Reviewed against the corpus (scratch candidate list, 305 pairs). Rules in merge_variants(); these are the cases
# rules cannot decide.
YEAR_TYPOS = {  # surface year that no such Act has -> the Act's real year
    "Hindu Marriage Act, 1956": "Hindu Marriage Act, 1955",
    "Explosive Act, 1984": "Explosives Act, 1884",
    "Narcotics Drugs and Psychotropic Substances Act, 1994": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "Identification of Prisoners Act, 1980": "Identification of Prisoners Act, 1920",
    "Registration of Births and Deaths Act, 1956": "Registration of Births and Deaths Act, 1969",
    "Registration of Birth and Deaths Act, 1956": "Registration of Births and Deaths Act, 1969",
    "Maharashtra Police Act, 1954": "Maharashtra Police Act, 1951",
    "Indian Arms Act, 1959": "Arms Act, 1959",
    "Code of Criminal Procedure Code, 1973": "Code of Criminal Procedure, 1973",
    "National Drugs and Psychotropic Substances Act, 1985": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "Narcotics Drugs and Psychotropic Substances Act, 1985": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "Indian Panal Code": "Indian Penal Code, 1860",
    "Indian Pen.al Code": "Indian Penal Code, 1860",
    "Indian D Penal Code": "Indian Penal Code, 1860",
    "E Indian Penal Code": "Indian Penal Code, 1860",
    "Security Exchange Board of India Act, 1992": "Securities and Exchange Board of India Act, 1992",
    "Food and Safety Standards Act, 2006": "Food Safety and Standards Act, 2006",
    "National Investigation Act, 2008": "National Investigation Agency Act, 2008",
}
NEVER = {  # similar names, different instruments
    frozenset({"GST Act, 2017", "CGST Act, 2017"}),
    frozenset({"Airport (Security) Rules, 2011", "Aircraft (Security) Rules, 2011"}),
    frozenset({"Banking Act", "Banning Act"}),
}
_KINDS_SET = {"act", "code", "sanhita", "adhiniyam", "rules", "regulations"}


def _words(n: str) -> list[str]:
    n = re.sub(r",?\s*\d{4}$", "", n).lower().replace("&", "and")
    return [w for w in re.sub(r"[^a-z0-9]+", " ", n).split() if w != "the"]


def _year(n: str) -> str | None:
    m = re.search(r"(\d{4})$", n)
    return m[1] if m else None


def merge_variants(usage: dict[str, int]) -> dict[str, str]:
    """name -> canonical name for spelling / OCR variants of one Act. usage = name -> mentions.
    Same kind (Act vs Rules) always. Different years never merge (Limitation Act 1908 / 1963 are two Acts),
    except YEAR_TYPOS. Short abbreviations ('TP Act') merge only on identical letters. An undated name joins a
    dated one only when the corpus has a single year for that name."""
    from rapidfuzz import fuzz

    out = {a: b for a, b in YEAR_TYPOS.items() if a in usage}
    names = sorted((n for n in usage if n not in out), key=lambda n: -usage[n])
    info = {n: (" ".join(_words(n)), "".join(_words(n)), _year(n),
                next((w for w in reversed(_words(n)) if w in _KINDS_SET), None)) for n in names}
    years_of = {}
    for n in names:
        if info[n][2]:
            years_of.setdefault(info[n][0], set()).add(info[n][2])
    roots: list[str] = []
    for a in names:
        ka, ca, ya, kinda = info[a]
        best = None
        for b in roots:
            kb, cb, yb, kindb = info[b]
            if kinda != kindb or frozenset({a, b}) in NEVER or (ya and yb and ya != yb):
                continue
            if min(len(ca), len(cb)) <= 8:
                ok = ca == cb
            else:
                s = fuzz.ratio(ka, kb)
                if ya and yb:
                    ok = s >= 92
                elif ya or yb:
                    dated = ka if ya else kb
                    ok = s >= 94 and len(years_of.get(dated, ())) == 1
                else:
                    ok = s >= 94
            if ok:
                best = b
                break
        if best is None:
            roots.append(a)
        else:
            out[a] = best
    # prefer the dated spelling as the canonical name of a group
    for root in roots:
        group = [n for n, r in out.items() if r == root] + [root]
        dated = [n for n in group if _year(n)]
        if not _year(root) and dated:
            new = max(dated, key=lambda n: usage[n])
            for n in group:
                out[n] = new
            out.pop(new, None)
    return {a: b for a, b in out.items() if a != b}
