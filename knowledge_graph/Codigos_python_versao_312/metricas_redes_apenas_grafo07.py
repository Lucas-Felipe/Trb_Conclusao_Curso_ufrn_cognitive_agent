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
    "codigocivil08"
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
            # Testa se os nós têm o campo `id`
            test_id = session.run("""
                MATCH (n)-[r]->(m)
                RETURN n.id AS source, m.id AS target
                LIMIT 1
            """).single()

            if test_id and test_id["source"] is not None and test_id["target"] is not None:
                query = "MATCH (n)-[r]->(m) RETURN n.id AS source, m.id AS target"
            else:
                # Tenta `name`
                test_name = session.run("""
                    MATCH (n)-[r]->(m)
                    RETURN n.nome AS source, m.nome AS target
                    LIMIT 1
                """).single()

                if test_name and test_name["source"] is not None and test_name["target"] is not None:
                    query = "MATCH (n)-[r]->(m) RETURN n.nome AS source, m.nome AS target"
                else:
                    # Fallback: usa elementId (único garantido)
                    query = "MATCH (n)-[r]->(m) RETURN elementId(n) AS source, elementId(m) AS target"

            result = session.run(query)
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

        # Eccentricity (apenas para maior componente conexa)
        try:
            largest_cc = max(nx.connected_components(UG), key=len)
            subgraph = UG.subgraph(largest_cc).copy()
            eccentricity = nx.eccentricity(subgraph)
            for node, val in eccentricity.items():
                node_data.setdefault(node, {})["eccentricity"] = val
        except Exception:
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
                diameter = nx.diameter(subgraph)
                radius = nx.radius(subgraph)
                center = nx.center(subgraph)
                periphery = nx.periphery(subgraph)

                writer.writerow(["Diâmetro", diameter])
                writer.writerow(["Raio", radius])
                writer.writerow(["Centro", ', '.join(center)])
                writer.writerow(["Periferia", ', '.join(periphery)])
            except Exception as e:
                writer.writerow(["Atenção", f"Métricas globais não calculadas: {str(e)}"])

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
