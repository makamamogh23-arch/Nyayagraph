"""data/graph/*.parquet -> Neo4j (idempotent MERGE, batched UNWIND).

Tables from the fine-tuned NER pass (ner/corpus_entities.py: Lawyer, lower Court, Witness) are loaded when present.

Connection: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD from env or .env (see graph/conn.py).
Usage: python -m graph.load [--wipe] [--only TABLE ...]
"""
import argparse
import os

import pandas as pd

from graph.conn import driver

G = "data/graph"
BATCH = 5000

CONSTRAINTS = [
    "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (n:Case) REQUIRE n.case_id IS UNIQUE",
    "CREATE CONSTRAINT para_id IF NOT EXISTS FOR (n:Paragraph) REQUIRE n.pid IS UNIQUE",
    "CREATE CONSTRAINT prov_id IF NOT EXISTS FOR (n:Provision) REQUIRE n.pid IS UNIQUE",
    "CREATE CONSTRAINT act_name IF NOT EXISTS FOR (n:Act) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT judge_key IF NOT EXISTS FOR (n:Judge) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT court_name IF NOT EXISTS FOR (n:Court) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT lawyer_key IF NOT EXISTS FOR (n:Lawyer) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT witness_id IF NOT EXISTS FOR (n:Witness) REQUIRE n.wid IS UNIQUE",
    "CREATE INDEX case_year IF NOT EXISTS FOR (n:Case) ON (n.year)",
    "CREATE INDEX prov_section IF NOT EXISTS FOR (n:Provision) ON (n.act, n.section)",
    "CREATE FULLTEXT INDEX para_text IF NOT EXISTS FOR (n:Paragraph) ON EACH [n.text]",
]

LOADS = [
    ("nodes_court", "MERGE (n:Court {name: r.name})"),
    ("nodes_act", "MERGE (n:Act {name: r.name})"),
    ("nodes_judge", "MERGE (n:Judge {key: r.key}) SET n.name = r.name"),
    ("nodes_case", "MERGE (n:Case {case_id: r.case_id}) SET n += r"),
    ("nodes_provision", "MERGE (n:Provision {pid: r.pid}) SET n.act = r.act, n.section = r.section"),
    ("nodes_paragraph", "MERGE (n:Paragraph {pid: r.pid}) SET n += r"),
    ("edges_part_of", "MATCH (p:Provision {pid: r.pid}), (a:Act {name: r.act}) MERGE (p)-[:PART_OF]->(a)"),
    ("edges_maps_to", "MATCH (a:Provision {pid: r.src}), (b:Provision {pid: r.dst}) "
                      "MERGE (a)-[e:MAPS_TO]->(b) SET e.new_sub = r.new_sub, e.confidence = r.confidence"),
    ("edges_decided_by", "MATCH (c:Case {case_id: r.case_id}), (k:Court {name: r.court}) MERGE (c)-[:DECIDED_BY]->(k)"),
    ("edges_has_paragraph", "MATCH (c:Case {case_id: r.case_id}), (p:Paragraph {pid: r.pid}) MERGE (c)-[:HAS_PARAGRAPH]->(p)"),
    ("edges_authored", "MATCH (j:Judge {key: r.judge}), (c:Case {case_id: r.case_id}) MERGE (j)-[:AUTHORED]->(c)"),
    ("edges_sat_on", "MATCH (j:Judge {key: r.judge}), (c:Case {case_id: r.case_id}) MERGE (j)-[:SAT_ON]->(c)"),
    ("edges_cites", "MATCH (a:Case {case_id: r.src}), (b:Case {case_id: r.dst}) "
                    "MERGE (a)-[e:CITES]->(b) SET e.paras = r.paras, e.n = r.n, e.in_footnote = r.in_footnote, "
                    "e.supra_paras = coalesce(r.supra_paras, [])"),
    # fine-tuned NER (optional): edges carry source = the model that produced them
    ("nodes_lawyer", "MERGE (n:Lawyer {key: r.key}) SET n.name = r.name"),
    ("nodes_lower_court", "MERGE (n:Court {name: r.name})"),
    ("nodes_witness", "MERGE (n:Witness {wid: r.wid}) SET n.name = r.name, n.pw = r.pw, n.case_id = r.case_id"),
    ("edges_appeared_in", "MATCH (l:Lawyer {key: r.lawyer}), (c:Case {case_id: r.case_id}) "
                          "MERGE (l)-[e:APPEARED_IN]->(c) SET e.side = r.side, e.senior = r.senior, e.source = r.source"),
    ("edges_appeal_from", "MATCH (c:Case {case_id: r.case_id}), (k:Court {name: r.court}) "
                          "MERGE (c)-[e:APPEAL_FROM]->(k) SET e.bench = r.bench, e.date = r.date, "
                          "e.case_numbers = r.case_numbers, e.raw = r.raw, e.source = r.source"),
    ("edges_witness_in", "MATCH (w:Witness {wid: r.wid}), (c:Case {case_id: r.case_id}) "
                         "MERGE (w)-[e:WITNESS_IN]->(c) SET e.paras = r.paras, e.source = r.source"),
    ("edges_invokes", "MATCH (c:Case {case_id: r.case_id}), (p:Provision {pid: r.pid}) "
                      "MERGE (c)-[e:INVOKES]->(p) SET e.paras = r.paras, e.n = r.n, e.subsections = r.subsections"),
]


def records(df: pd.DataFrame) -> list[dict]:
    """parquet -> plain python values Neo4j accepts (lists, no NaN / numpy types)."""
    out = []
    for row in df.to_dict("records"):
        clean = {}
        for k, v in row.items():
            if hasattr(v, "tolist"):
                v = v.tolist()
            if isinstance(v, float) and v != v:
                continue
            clean[k] = v
        out.append(clean)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wipe", action="store_true", help="delete all nodes first")
    ap.add_argument("--only", nargs="+", help="load just these tables, e.g. nodes_lawyer edges_appeared_in")
    args = ap.parse_args()
    with driver() as d, d.session() as s:
        if args.wipe:
            s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
        for q in CONSTRAINTS:
            s.run(q)
        for name, cypher in LOADS:
            if args.only and name not in args.only:
                continue
            if not os.path.exists(f"{G}/{name}.parquet"):
                print(f"{name:22s} (absent, skipped)")
                continue
            rows = records(pd.read_parquet(f"{G}/{name}.parquet"))
            for i in range(0, len(rows), BATCH):
                s.run(f"UNWIND $rows AS r {cypher}", rows=rows[i:i + BATCH]).consume()
            print(f"{name:22s} {len(rows):>8,}")
        for label in ["Case", "Paragraph", "Provision", "Act", "Judge", "Court", "Lawyer", "Witness"]:
            print(label, s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"])
        for rel in ["CITES", "INVOKES", "PART_OF", "MAPS_TO", "AUTHORED", "SAT_ON", "DECIDED_BY", "HAS_PARAGRAPH",
                    "APPEARED_IN", "APPEAL_FROM", "WITNESS_IN"]:
            print(rel, s.run(f"MATCH ()-[e:{rel}]->() RETURN count(e) AS c").single()["c"])


if __name__ == "__main__":
    main()
