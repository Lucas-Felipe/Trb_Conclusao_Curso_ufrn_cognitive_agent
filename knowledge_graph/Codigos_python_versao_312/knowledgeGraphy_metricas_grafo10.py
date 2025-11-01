from neo4j import GraphDatabase
import networkx as nx
import csv
import os

# Configurações do Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # Substitua pela sua senha

# Banco de dados a ser analisado
databases = [
    "codigocivil10"
]

OUTPUT_DIR = "codigocivil10_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class GraphAnalyzer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def load_graph_from_database(self, db_name):
        """
        Carrega arestas do grafo completo do banco, filtrando
        apenas os relacionamentos relevantes do grafo jurídico.
        Usa coalesce para pegar id, nome ou elementId dos nós.
        """
        with self.driver.session(database=db_name) as session:
            query = """
            MATCH (n)-[r]->(m)
            WHERE type(r) IN [
                'TEM_LIVRO', 'TEM_TITULO', 'TEM_CAPITULO', 'TEM_SECAO',
                'TEM_ARTIGO', 'TEM_CHUNK', 'MENCIONA'
            ]
            RETURN
                coalesce(n.id, n.nome, elementId(n)) AS source,
                coalesce(m.id, m.nome, elementId(m)) AS target
            """
            result = session.run(query)
            edges = [
                (record["source"], record["target"])
                for record in result
                if record["source"] is not None and record["target"] is not None
            ]
            return edges

    def load_subgraph_chunks_entidades(self, db_name):
        """
        Carrega arestas do subgrafo Chunk -> Entidade para análise específica.
        """
        with self.driver.session(database=db_name) as session:
            query = """
            MATCH (c:Chunk)-[:MENCIONA]->(e:Entidade)
            RETURN
                coalesce(c.id, c.nome, elementId(c)) AS source,
                coalesce(e.id, e.nome, elementId(e)) AS target
            """
            result = session.run(query)
            edges = [
                (record["source"], record["target"])
                for record in result
                if record["source"] is not None and record["target"] is not None
            ]
            return edges

    def analyze_graph(self, edges, db_name, suffix=""):
        """
        Recebe arestas e calcula métricas do grafo.
        Salva CSV com nome contendo sufixo para diferenciar análises.
        """
        G = nx.DiGraph()
        G.add_edges_from(edges)
        UG = G.to_undirected()

        node_data = {}

        # Grau (degree)
        for node, degree in G.degree():
            node_data.setdefault(node, {})["degree"] = degree

        # Clustering (em grafo não-direcionado)
        clustering = nx.clustering(UG)
        for node, val in clustering.items():
            node_data.setdefault(node, {})["clustering"] = val

        # Closeness centrality
        closeness = nx.closeness_centrality(G)
        for node, val in closeness.items():
            node_data.setdefault(node, {})["closeness"] = val

        # Betweenness centrality
        betweenness = nx.betweenness_centrality(G)
        for node, val in betweenness.items():
            node_data.setdefault(node, {})["betweenness"] = val

        # Eccentricity (para maior componente conexa do grafo não direcionado)
        try:
            largest_cc = max(nx.connected_components(UG), key=len)
            subgraph = UG.subgraph(largest_cc).copy()
            eccentricity = nx.eccentricity(subgraph)
            for node, val in eccentricity.items():
                node_data.setdefault(node, {})["eccentricity"] = val
        except Exception:
            eccentricity = {}

        # Salvar CSV
        output_path = os.path.join(OUTPUT_DIR, f"{db_name}_metrics{suffix}.csv")
        with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "node", "degree", "clustering",
                "closeness_centrality", "betweenness_centrality", "eccentricity"
            ])

            for node, metrics in node_data.items():
                writer.writerow([
                    node,
                    metrics.get("degree", ""),
                    metrics.get("clustering", ""),
                    metrics.get("closeness", ""),
                    metrics.get("betweenness", ""),
                    metrics.get("eccentricity", "")
                ])

            # Métricas globais do maior componente conexo
            writer.writerow([])
            writer.writerow(["Métricas globais"])

            try:
                diameter = nx.diameter(subgraph)
                radius = nx.radius(subgraph)
                center = nx.center(subgraph)
                periphery = nx.periphery(subgraph)

                writer.writerow(["Diâmetro", diameter])
                writer.writerow(["Raio", radius])
                writer.writerow(["Centro", ', '.join(map(str, center))])
                writer.writerow(["Periferia", ', '.join(map(str, periphery))])
            except Exception as e:
                writer.writerow(["Atenção", f"Métricas globais não calculadas: {str(e)}"])

        print(f"✔️ Métricas salvas em: {output_path}")

    def run(self):
        for db in databases:
            print(f"\n=== Analisando banco: {db} ===")

            # Grafo completo com relacionamentos jurídicos
            edges = self.load_graph_from_database(db)
            self.analyze_graph(edges, db, suffix="_completo")

            # Exemplo: análise só do subgrafo Chunk -> Entidade
            edges_chunks_entidades = self.load_subgraph_chunks_entidades(db)
            self.analyze_graph(edges_chunks_entidades, db, suffix="_chunks_entidades")


if __name__ == "__main__":
    analyzer = GraphAnalyzer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        analyzer.run()
    finally:
        analyzer.close()
