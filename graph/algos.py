"""GDS on the citation graph: PageRank (directed CITES) and Louvain (undirected CITES), written back to :Case.

Requires the Graph Data Science plugin. Usage: python -m graph.algos
"""
from graph.conn import driver

PROJ = "cites"


def main():
    with driver() as d, d.session() as s:
        s.run("CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=PROJ).consume()
        s.run("""CALL gds.graph.project($g, 'Case',
                 {CITES: {orientation: 'NATURAL'}, CITES_U: {type: 'CITES', orientation: 'UNDIRECTED'}})""",
              g=PROJ).consume()
        pr = s.run("""CALL gds.pageRank.write($g, {relationshipTypes: ['CITES'], writeProperty: 'pagerank',
                      dampingFactor: 0.85, maxIterations: 50}) YIELD nodePropertiesWritten, ranIterations
                      RETURN nodePropertiesWritten, ranIterations""", g=PROJ).single()
        lv = s.run("""CALL gds.louvain.write($g, {relationshipTypes: ['CITES_U'], writeProperty: 'community'})
                      YIELD communityCount, modularity RETURN communityCount, modularity""", g=PROJ).single()
        s.run("CALL gds.graph.drop($g) YIELD graphName RETURN graphName", g=PROJ).consume()
        print(f"pagerank: {pr['nodePropertiesWritten']} nodes, {pr['ranIterations']} iterations")
        print(f"louvain: {lv['communityCount']} communities, modularity {lv['modularity']:.3f}")
        print("top cases by pagerank:")
        for r in s.run("""MATCH (c:Case) RETURN c.case_id AS id, c.title AS title, c.keys AS keys,
                          round(c.pagerank, 3) AS pr, COUNT { (c)<-[:CITES]-() } AS cited_by
                          ORDER BY pr DESC LIMIT 10"""):
            print(f"  {r['pr']:>7} cited_by={r['cited_by']:>4}  {r['title'] or ', '.join(r['keys'][:2])}")
        print("largest communities (top provisions invoked by member judgments):")
        for r in s.run("""MATCH (c:Case) WITH c.community AS com, count(*) AS size ORDER BY size DESC LIMIT 6
                          MATCH (m:Case {community: com})-[:INVOKES]->(p:Provision)
                          WITH com, size, p, count(*) AS k ORDER BY k DESC
                          RETURN com, size, collect(p.act + ' s.' + p.section)[..4] AS provisions"""):
            print(f"  community {r['com']} size={r['size']}: {r['provisions']}")


if __name__ == "__main__":
    main()
