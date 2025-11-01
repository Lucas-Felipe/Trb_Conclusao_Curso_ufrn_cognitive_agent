import os
import re
import uuid
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.prompts import PromptTemplate
from langchain_community.graphs.graph_document import Node, Relationship
from neo4j import GraphDatabase

load_dotenv()

# ——— Carregar documentos ———
DOCS_PATH = "C:/Users/LUCAS FELIPE/Desktop/knowledge_graph/cod_juri"
loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# ——— Limpar grafo anterior ———
uri = os.getenv('NEO4J_URI')
username = os.getenv('NEO4J_USERNAME')
password = os.getenv('NEO4J_PASSWORD')
database = "codigocivil11"

driver = GraphDatabase.driver(uri, auth=(username, password))

def limpar_grafo(tx):
    tx.run("MATCH (n) DETACH DELETE n")

with driver.session(database=database) as session:
    session.execute_write(limpar_grafo)

# ——— Funções de limpeza e chunking ———
def limpar_texto(texto):
    texto = ''.join(c for c in texto if c.isprintable() or c in '\n\r\t§ºª\u201c\u201d\u2013\u2014')
    texto = re.sub(r"https?://\S+", "", texto)
    texto = re.sub(r"^\s*\.", "", texto)
    texto = re.sub(r"\b\d{2}/\d{2}/\d{4}, \d{2}:\d{2} L\d+\b", "", texto)
    texto = re.sub(r"\b\d+/\d+\b", "", texto)
    linhas = texto.splitlines()
    linhas_limpas = [re.sub(r'\s+', ' ', linha).strip() for linha in linhas if linha.strip()]
    return "\n".join(linhas_limpas).strip()

def remove_parte_final(texto):
    padrao = r"Brasília,\s+\d{1,2} de [A-Za-z]+ de \d{4}.*?FERNANDO HENRIQUE CARDOSO"
    match = re.search(padrao, texto, flags=re.DOTALL | re.IGNORECASE)
    return texto[:match.start()].strip() if match else texto

def dividir_em_chunks_estruturados(texto):
    padrao = r"(?=Art\.?\s*\d+(?:º|\.)?)"
    matches = list(re.finditer(padrao, texto))
    chunks = []
    if matches and matches[0].start() > 0:
        intro = texto[:matches[0].start()].strip()
        if len(intro) > 50:
            chunks.append(intro)
    for i, m in enumerate(matches):
        inicio = m.start()
        fim = matches[i+1].start() if i+1 < len(matches) else len(texto)
        trecho = texto[inicio:fim].strip()
        if len(trecho) > 50:
            chunks.append(trecho)
    return chunks

# ——— Pré-processamento dos documentos ———
all_chunks = []
for doc in docs:
    texto = doc.page_content
    clean = limpar_texto(texto)
    clean = remove_parte_final(clean)
    chunks = dividir_em_chunks_estruturados(clean)
    for chunk in chunks:
        all_chunks.append(Document(page_content=chunk, metadata={"source": doc.metadata.get("source", "")}))

# ——— Setup embeddings e grafo ———
embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
)

graph = Neo4jGraph(
    url=uri,
    username=username,
    password=password,
    database=database
)

# ——— Criar embeddings para os chunks ———
for chunk in all_chunks:
    chunk_id = str(uuid.uuid4())
    emb = embedding_provider.embed_query(chunk.page_content)
    props = {
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": emb,
        "source": chunk.metadata.get("source", "")
    }
    graph.query("""
        MERGE (chunk:Chunk {id: $chunk_id})
        SET chunk.text = $text
        WITH chunk
        CALL db.create.setNodeVectorProperty(chunk, 'textEmbedding', $embedding)
    """, props)

# ——— Criar índice vetorial ———
graph.query("DROP INDEX chunkVector IF EXISTS")
graph.query("""
CREATE VECTOR INDEX chunkVector
FOR (c:Chunk) ON (c.textEmbedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
""")

# ——— LLM + Prompt para extrair entidades/relacionamentos ———
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="gpt-3.5-turbo"
)

prompt = PromptTemplate.from_template("""
Você é um especialista em Direito brasileiro e extração de grafos semânticos a partir de textos legislativos.

A partir do texto jurídico abaixo, extraia as **entidades jurídicas relevantes** e os **relacionamentos explícitos ou implícitos** entre elas.

⚖️ Considere como entidades:
- Sujeitos de direito (ex: Pessoa Natural, Pessoa Jurídica, Autoridade Pública)
- Instituições ou órgãos (ex: União, Senado, Ministério Público)
- Atores normativos (ex: Juiz, Parte, Inventariante, Herdeiro)
- Conceitos jurídicos (ex: Domicílio, Capacidade Civil, Testamento)
- Obrigações, direitos, deveres ou sanções
- Prazos e valores legais (ex: 30 dias, 100 salários mínimos)

🔗 Para os relacionamentos, identifique:
- Quem exerce obrigações sobre quem
- Regras de competência ou subordinação
- Relações de causa e efeito (ex: "Se não for cumprido, haverá multa")
- Relações de exceção, condição ou consequência

📄 Texto:
{input}

Responda no formato JSON com os seguintes campos:
{{
  "nodes": [
    {{"id": "1", "name": "Pessoa Natural", "type": "Pessoa"}},
    {{"id": "2", "name": "Testamento", "type": "Instituto Jurídico"}},
    ...
  ],
  "relationships": [
    {{"source": "1", "target": "2", "type": "TEM_DIREITO_SOBRE"}},
    ...
  ]
}}
""")

doc_transformer = LLMGraphTransformer(llm=llm, prompt=prompt)

# ——— Gerar e inserir entidades e relações no grafo ———
for chunk in all_chunks:
    chunk_id = str(uuid.uuid4())
    chunk.metadata["chunk_id"] = chunk_id
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    for gd in graph_docs:
        nodes_corrigidos = []
        for node in gd.nodes:
            nome_valor = node.properties.get("name", "").strip()
            if not nome_valor:
                continue
            label = node.type.strip().title().replace(" ", "_")
            novo_node = Node(
                id=node.id,
                type=label,
                properties={"name": nome_valor}
            )
            nodes_corrigidos.append(novo_node)

        rels_corrigidas = []
        for rel in gd.relationships:
            source = rel.source
            target = rel.target
            if isinstance(source, Node) and isinstance(target, Node):
                new_rel = Relationship(
                    source=Node(id=source.id, type=source.type.strip().title().replace(" ", "_")),
                    target=Node(id=target.id, type=target.type.strip().title().replace(" ", "_")),
                    type=rel.type.upper().replace(" ", "_")
                )
                rels_corrigidas.append(new_rel)

        chunk_node = Node(id=chunk_id, type="Chunk", properties={"text": chunk.page_content})
        for node in nodes_corrigidos:
            rels_corrigidas.append(Relationship(source=chunk_node, target=node, type="HAS_ENTITY"))

        gd_corrigido = type(gd)(
            nodes=nodes_corrigidos,
            relationships=rels_corrigidas,
            source=chunk  # chunk é um Document válido
        )

        graph.add_graph_documents([gd_corrigido])
