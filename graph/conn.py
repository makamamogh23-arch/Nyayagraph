"""Neo4j connection settings from env, falling back to the project's .env (gitignored)."""
import os

from neo4j import GraphDatabase


def _dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            k, sep, v = line.strip().partition("=")
            if sep and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def driver():
    _dotenv()
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    return GraphDatabase.driver(uri, auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]))
