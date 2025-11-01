from neo4j import GraphDatabase
import networkx as nx
import csv
import os

# Configuração do Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # Substitua pela sua senha

# Bancos de dados a serem analisados
databases = [
    "codigocivil02", "codigocivil03", "codigocivil04",
    "codigocivil05", "codigocivil07"
]

OUTPUT_DIR = "metricas_csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class GraphAnalyzer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def load_graph_from_database(self, db_name):
        with self.driver.session(database=db_name) as session:
            result = session.run("""
                MATCH (n)-[r]->(m)
                RETURN n.id AS source, m.id AS target
            """)
            edges = [
                (record["source"], record["target"])
                for record in result
                if record["source"] is not None and record["target"] is not None
            ]
            return edges

    def analyze_graph(self, edges, db_name):
        G = nx.DiGraph()
        G.add_edges_from(edges)
        UG = G.to_undirected()

        node_data = {}

        # Grau
        for node, degree in G.degree():
            node_data.setdefault(node, {})["degree"] = degree

        # Clustering
        clustering = nx.clustering(UG)
        for node, val in clustering.items():
            node_data.setdefault(node, {})["clustering"] = val

        # Closeness
        closeness = nx.closeness_centrality(G)
        for node, val in closeness.items():
            node_data.setdefault(node, {})["closeness"] = val

        # Betweenness
        betweenness = nx.betweenness_centrality(G)
        for node, val in betweenness.items():
            node_data.setdefault(node, {})["betweenness"] = val

        # Eccentricity (pode falhar se o grafo for desconexo)
        try:
            eccentricity = nx.eccentricity(UG)
            for node, val in eccentricity.items():
                node_data.setdefault(node, {})["eccentricity"] = val
        except nx.NetworkXError:
            eccentricity = {}

        # Salva dados por nó em CSV
        output_path = os.path.join(OUTPUT_DIR, f"{db_name}_metrics.csv")
        with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["node", "degree", "clustering", "closeness_centrality", "betweenness_centrality", "eccentricity"])

            for node, metrics in node_data.items():
                writer.writerow([
                    node,
                    metrics.get("degree", ""),
                    metrics.get("clustering", ""),
                    metrics.get("closeness", ""),
                    metrics.get("betweenness", ""),
                    metrics.get("eccentricity", "")
                ])

            # Métricas globais
            writer.writerow([])
            writer.writerow(["Métricas globais"])

            try:
                diameter = nx.diameter(UG)
                radius = nx.radius(UG)
                center = nx.center(UG)
                periphery = nx.periphery(UG)

                writer.writerow(["Diâmetro", diameter])
                writer.writerow(["Raio", radius])
                writer.writerow(["Centro", ', '.join(center)])
                writer.writerow(["Periferia", ', '.join(periphery)])
            except nx.NetworkXError:
                writer.writerow(["Atenção", "Métricas globais não calculadas: grafo desconexo."])

        print(f"✔️ Métricas salvas em: {output_path}")

    def run(self):
        for db in databases:
            print(f"\n=== Analisando banco: {db} ===")
            edges = self.load_graph_from_database(db)
            self.analyze_graph(edges, db)

if __name__ == "__main__":
    analyzer = GraphAnalyzer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        analyzer.run()
    finally:
        analyzer.close()
