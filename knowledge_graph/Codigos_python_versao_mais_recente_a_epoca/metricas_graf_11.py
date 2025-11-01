import os
import networkx as nx
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = "codigocivil11"  # altere se estiver usando outro nome

# Conectar ao Neo4j
driver = GraphDatabase.driver(uri, auth=(username, password))

def extrair_grafo():
    G = nx.DiGraph()
    with driver.session(database=database) as session:
        # Nós
        nodes = session.run("MATCH (n) RETURN id(n) as id, labels(n) as labels, n.name as name")
        for record in nodes:
            node_id = record["id"]
            label = record["labels"][0] if record["labels"] else "NoLabel"
            name = record["name"] or f"{label}_{node_id}"
            G.add_node(node_id, label=label, name=name)

        # Relacionamentos
        rels = session.run("MATCH (a)-[r]->(b) RETURN id(a) as source, id(b) as target, type(r) as type")
        for record in rels:
            G.add_edge(record["source"], record["target"], type=record["type"])
    return G

def calcular_metricas(G):
    print("Calculando métricas do grafo...")
    grau_total = dict(G.degree())
    grau_entrada = dict(G.in_degree())
    grau_saida = dict(G.out_degree())
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G.to_undirected())

    eccentricity = {}
    radius = diameter = None
    periphery = []

    if nx.is_connected(G.to_undirected()):
        eccentricity = nx.eccentricity(G)
        radius = nx.radius(G)
        diameter = nx.diameter(G)
        periphery = nx.periphery(G)
    else:
        print("⚠️ Grafo desconexo — ignorando eccentricity, radius, diameter e periphery.")

    df = pd.DataFrame({
        "node_id": list(G.nodes),
        "label": [G.nodes[n].get("label", "") for n in G.nodes],
        "name": [G.nodes[n].get("name", "") for n in G.nodes],
        "grau_total": [grau_total.get(n, 0) for n in G.nodes],
        "grau_entrada": [grau_entrada.get(n, 0) for n in G.nodes],
        "grau_saida": [grau_saida.get(n, 0) for n in G.nodes],
        "closeness": [closeness.get(n, 0) for n in G.nodes],
        "betweenness": [betweenness.get(n, 0) for n in G.nodes],
        "clustering": [clustering.get(n, 0) for n in G.nodes],
        "eccentricity": [eccentricity.get(n, None) for n in G.nodes],
        "periphery": [n in periphery for n in G.nodes]
    })

    df.to_csv("metricas_grafo_novo.csv", index=False)
    print("✅ Métricas salvas em metricas_grafo.csv")

    if radius is not None and diameter is not None:
        print(f"\n📏 Radius: {radius}")
        print(f"📏 Diameter: {diameter}")

def main():
    G = extrair_grafo()
    calcular_metricas(G)

if __name__ == "__main__":
    main()
