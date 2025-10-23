from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "your_password")

driver = GraphDatabase.driver(uri, auth=(user, password))

def run_cypher(query, params=None):
    with driver.session() as session:
        result = session.run(query, params)
        return [r.data() for r in result]

def get_crop_related_info(crop_name):
    """
    Returns domain-wise info for a given crop.
    """
    query = """
    MATCH (c:Crop {name: $crop_name})-[:HAS_INFO]->(i)
    RETURN i.domain AS domain, i.description AS description
    """
    records = run_cypher(query, {"crop_name": crop_name})
    if not records:
        return f"No structured graph info found for {crop_name}."
    return "\n".join([f"{r['domain']}: {r['description']}" for r in records])
