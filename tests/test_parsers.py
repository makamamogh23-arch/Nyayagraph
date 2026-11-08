"""Regression tests for the text parsers and mappings. Each case is behaviour that was checked by hand; several pin
bugs fixed during the project (noted inline). No network, no Neo4j, no model calls.

Run: python -m pytest -q
"""
import pandas as pd
import pytest

from api.server import display_key, display_title
from eval.rag_bench import new_act_maps, new_act_query
from link.acts import YEAR_TYPOS, merge_variants, normalize, strip_prefix
from link.citations import extract as cites
from link.law_map import CHECKS
from link.provisions import IPC_ONLY, extract, fix_ipc_ocr
from link.resolve import name_before, parties
from rag.retrieve import DEFINED_BY, DEFINITION_OF


def provs(text: str) -> list[tuple[str | None, str]]:
    return [(r["act"], r["section"]) for r in extract(text, {})]


IPC, CRPC, IEA = "Indian Penal Code, 1860", "Code of Criminal Procedure, 1973", "Indian Evidence Act, 1872"
BNS, BNSS, BSA = "Bharatiya Nyaya Sanhita, 2023", "Bharatiya Nagarik Suraksha Sanhita, 2023", "Bharatiya Sakshya Adhiniyam, 2023"


# --- citations -------------------------------------------------------------------------------------------------

def test_citation_reporters_and_parallel_groups():
    t = ("relied on State of U.P. v. Satish, (2005) 3 SCC 114 and AIR 1953 SC 364 : 1954 SCR 145; see also "
         "2023 INSC 239 and 2019 SCC OnLine SC 1346, [2016] 8 S.C.R. 380")
    hits = cites(t)
    assert [h["key"] for h in hits] == ["SCC|2005|3|114", "AIR SC|1953||364", "SCR|1954||145", "INSC|2023||239",
                                        "SCC OnLine SC|2019||1346", "SCR|2016|8|380"]
    groups = [h["group"] for h in hits]
    assert groups[1] == groups[2]          # "AIR ... : SCR ..." is one case cited in two reporters
    assert len(set(groups)) == 5


def test_name_before_citation():
    t = "In Sharad Birdhichand Sarda v. State of Maharashtra, (1984) 4 SCC 116, the Court"
    assert name_before(t, t.index("(1984)")) == "Sharad Birdhichand Sarda v. State of Maharashtra"
    assert parties("Sharad Birdhichand Sarda v. State of Maharashtra") == ["sharad birdhichand sarda", "maharashtra"]


# --- provisions ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Sections 302 and 34 IPC", [(IPC, "302"), (IPC, "34")]),
    ("Section 498A IPC", [(IPC, "498A")]),
    ("Section 302 of the IPC and Section 25 of the Arms Act", [(IPC, "302"), ("Arms Act, 1959", "25")]),
    # a spaced section suffix belongs to the section, not the Act name ("A of the Code")
    ("acquitted under Section 498 A of the Indian Penal Code, that", [(IPC, "498A")]),
    # Part / Chapter qualifiers are skipped, the Act after them is kept
    ("Section 304, Part II of the Indian Penal Code.", [(IPC, "304")]),
    ("Section 304 Part I of the Indian Penal Code", [(IPC, "304")]),
    # BNSS must not be read as BNS (ordered regex alternation)
    ("Section 528 BNSS", [(BNSS, "528")]),
    ("Section 483 of the BNSS", [(BNSS, "483")]),
    ("Section 35 BNSS read with Section 3(5) BNS", [(BNSS, "35"), (BNS, "3")]),
    ("Section 103 BNS", [(BNS, "103")]),
])
def test_provision_parsing(text, expected):
    assert provs(text) == expected


def test_chapter_qualifier_range():
    got = provs("Sections 149 to 152 of Chapter XI of the Code")
    assert sorted(s for _, s in got) == ["149", "150", "151", "152"]
    assert {a for a, _ in got} == {CRPC}   # bare "the Code" defaults to CrPC


def test_ipc_ocr_fixes():
    df = pd.DataFrame({"act": [IPC, IPC, None, CRPC], "section": ["1208", "304I", "3048", "40I"]})
    out = fix_ipc_ocr(df)["section"].tolist()
    assert out == ["120B", "304", "304B", "401"]   # 120B/304B from OCR "8"; Part I of 304; "40 I" = 401


def test_ipc_only_sections_are_lettered_ipc_sections():
    assert {"498A", "304B", "120B", "376D"} <= IPC_ONLY
    assert not IPC_ONLY & {"41A", "164A", "436A", "561A"}   # CrPC lettered sections


# --- act names -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, canon", [
    ("IPC", IPC),
    ("A of the Indian Penal Code", IPC),
    ("Part-I of G the Indian Penal Code", IPC),
    ("E Indian Penal Code", IPC),
    ("(C) of the Narcotic Drugs and Psychotropic Substances Act, 1985", "Narcotic Drugs and Psychotropic Substances Act, 1985"),
    ("In the Negotiable Instruments Act, 1881", "Negotiable Instruments Act, 1881"),
    ("SC/ST (POA) Act", "Scheduled Castes and the Scheduled Tribes (Prevention of Atrocities) Act, 1989"),
    ("Terrorist and Disruptive Act", "Terrorist and Disruptive Activities (Prevention) Act, 1987"),
    ("UP Gangsters Act", "Uttar Pradesh Gangsters and Anti-Social Activities (Prevention) Act, 1986"),
    ("Arms Act", "Arms Act, 1959"),
    ("UK Sexual Offences Act, 2003", "UK Sexual Offences Act, 2003"),
    ("J and K Juvenile Justice (Care and Protection of Children) Act, 2013",
     "J and K Juvenile Justice (Care and Protection of Children) Act, 2013"),
])
def test_normalize(raw, canon):
    assert normalize(raw) == canon


def test_anaphora_is_not_an_act():
    assert normalize("Chapter XVI of the Code") is None
    assert normalize("Sections of the Act") is None
    assert strip_prefix("Order (IV) of the Supreme Court Rules, 2013") == "Supreme Court Rules, 2013"


def test_merge_variants_rules():
    usage = {"Indian Penal Code, 1860": 1000, "Jndian Penal Code, 1860": 5, "India Penal Code": 3,
             "Limitation Act, 1963": 4, "Limitation Act, 1908": 2,
             "Protection of Women from Domestic Violence Act, 2005": 11,
             "Protection of Women from Domestic Violence Rules, 2006": 2,
             "PO Act": 4, "PoC Act": 2, "Hindu Marriage Act, 1956": 3, "Hindu Marriage Act, 1955": 11}
    m = merge_variants(usage)
    assert m["Jndian Penal Code, 1860"] == "Indian Penal Code, 1860"
    assert m["India Penal Code"] == "Indian Penal Code, 1860"           # undated joins the only dated version
    assert "Limitation Act, 1908" not in m                              # different years are different Acts
    assert "Protection of Women from Domestic Violence Rules, 2006" not in m   # Rules never merge into an Act
    assert "PoC Act" not in m and "PO Act" not in m                     # abbreviations need identical letters
    assert m["Hindu Marriage Act, 1956"] == YEAR_TYPOS["Hindu Marriage Act, 1956"]


# --- old -> new section tables ----------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["crpc_bnss", "iea_bsa"])
def test_law_map_checks(name):
    df = pd.read_csv(f"data/ref/{name}.csv", dtype=str).fillna("")
    for old, new in CHECKS[name].items():
        assert new in set(df.loc[df["old"] == old, "new"]), f"{name}: {old} -> {new}"


def test_iea_27_is_proviso_to_bsa_23():
    df = pd.read_csv("data/ref/iea_bsa.csv", dtype=str).fillna("")
    row = df[df["old"] == "27"].iloc[0]
    assert (row["new"], row["new_sub"]) == ("23", "proviso")


@pytest.mark.parametrize("ipc, bns", [("302", "103"), ("300", "101"), ("299", "100"), ("304B", "80"), ("498A", "85"),
                                      ("420", "318"), ("34", "3"), ("120B", "61")])
def test_ipc_bns_core(ipc, bns):
    df = pd.read_csv("data/ref/ipc_bns.csv", dtype=str).fillna("")
    assert bns in set(df.loc[df["ipc"] == ipc, "bns"])


def test_definition_companions_are_labelled_and_mapped():
    df = pd.read_csv("data/ref/ipc_bns.csv", dtype=str).fillna("")
    mapped = set(df["ipc"])
    for punish, defs in DEFINED_BY.items():
        for d in defs:
            assert d in DEFINITION_OF, d
            assert d in mapped, f"IPC {d} (defines {punish}) has no BNS counterpart"


def test_new_act_query_rewrite():
    q = "Does a mere breach of a loan promise constitute criminal breach of trust under Section 405 IPC?"
    assert new_act_query(q, new_act_maps()).endswith("under Section 316 of the BNS?")
    assert new_act_query("What is the test for bail?", new_act_maps()) is None


# --- display ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, shown", [
    ("UDAYAKUMAR versus STATE OF TAMIL NADU", "Udayakumar v. State of Tamil Nadu"),
    ("STATE OF U.P. versus VEERPAL & ANR.", "State of U.P. v. Veerpal & Anr."),
    ("STATE (GOVT. OF NCT OF DELHI) versus X", "State (Govt. of NCT of Delhi) v. X"),
    ("K.P. TAMILMARAN versus THE STATE BY DEPUTY SUPERINTENDENT OF POLICE",
     "K.P. Tamilmaran v. The State by Deputy Superintendent of Police"),
    ("Sharad Birdhichand Sarda v. State of Maharashtra", "Sharad Birdhichand Sarda v. State of Maharashtra"),
])
def test_display_title(raw, shown):
    assert display_title(raw) == shown


@pytest.mark.parametrize("key, shown", [
    ("SCC|2013|7|211", "(2013) 7 SCC 211"), ("AIR SC|1977||45", "AIR 1977 SC 45"), ("SCR|1962|3|10", "[1962] 3 SCR 10"),
    ("SCC OnLine SC|2019||1346", "2019 SCC OnLine SC 1346"), ("SCC (Cri)|2008|3|858", "(2008) 3 SCC (Cri) 858"),
    ("AIR PC|1944||73", "AIR 1944 PC 73"),
])
def test_display_key(key, shown):
    assert display_key(key) == shown
