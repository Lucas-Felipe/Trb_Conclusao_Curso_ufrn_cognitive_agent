import os
import re
import hashlib
import unicodedata
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.prompts import PromptTemplate
from langchain_community.graphs.graph_document import Node, Relationship

load_dotenv()

# --- Carregamento dos documentos ---
DOCS_PATH = "C:/Users/LUCAS FELIPE/Desktop/knowledge_graph/cod_juri"
loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()


def limpeza_antes_do_split(texto: str) -> str:
    linhas = texto.splitlines()
    linhas_limpa = []
    for linha in linhas:
        if re.match(r'^\d{2}/\d{2}/\d{4}, \d{2}:\d{2} L\d+', linha):
            continue
        if re.search(r'https?://\S+|www\.\S+', linha):
            continue
        if re.search(r'\bíndice\b', linha, re.IGNORECASE):
            continue
        linha = unicodedata.normalize('NFKD', linha).encode('ASCII', 'ignore').decode('utf-8', 'ignore')
        linhas_limpa.append(linha.strip())
    return '\n'.join(linhas_limpa)

def gerar_hash(texto):
    return hashlib.md5(texto.encode()).hexdigest()

def split_legal_text(text):
    artigo_pattern = r"(?=Art\.?\s*\d+[ºA-Za-z\-]*)"
    artigos = re.split(artigo_pattern, text, flags=re.MULTILINE)

    current_livro = ""
    current_titulo = ""
    current_capitulo = ""
    documents = []

    for raw in artigos:
        if not raw.strip():
            continue

        if re.match(r"^\s*LIVRO", raw, re.IGNORECASE):
            current_livro = raw.strip()
            continue
        if re.match(r"^\s*T[ÍI]TULO", raw, re.IGNORECASE):
            current_titulo = raw.strip()
            continue
        if re.match(r"^\s*CAP[ÍI]TULO", raw, re.IGNORECASE):
            current_capitulo = raw.strip()
            continue

        match_art = re.match(r"^\s*Art\.?\s*(\d+[ºo]?)", raw)
        artigo_num = match_art.group(1) if match_art else "desconhecido"
        artigo_completo = "Art. " + raw.strip()

        paragrafos = re.split(r"(?=^\s*(§\s*\d+º|Parágrafo único))", artigo_completo, flags=re.MULTILINE)
        partes_artigo = [paragrafos[0].strip()]
        for i in range(1, len(paragrafos), 2):
            paragrafo = paragrafos[i] + paragrafos[i + 1] if (i + 1) < len(paragrafos) else paragrafos[i]
            partes_artigo.append(paragrafo.strip())

        for parte in partes_artigo:
            incisos = re.split(r"(?=^\s*[IVXLCDM]+\s*-\s+)", parte, flags=re.MULTILINE)
            for inciso in incisos:
                if inciso.strip():
                    chunk_text = inciso.strip()
                    chunk_id = f"{artigo_num}_{gerar_hash(chunk_text)}"
                    documents.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "livro": current_livro,
                            "titulo": current_titulo,
                            "capitulo": current_capitulo,
                            "artigo": f"Art. {artigo_num}",
                            "chunk_id": chunk_id,
                            "source": "codigo_civil.pdf",
                            "page": 0
                        }
                    ))
    return documents

all_chunks = []
for doc in docs:
    texto_base = limpeza_antes_do_split(doc.page_content)
    chunks = split_legal_text(texto_base)
    for chunk in chunks:
        chunk.page_content = re.sub(r'\s+', ' ', chunk.page_content).strip()
        all_chunks.append(chunk)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    database="codigocivil06"
)

for chunk in all_chunks:
    chunk_id = chunk.metadata["chunk_id"]
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    properties = {
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding,
        "livro": chunk.metadata.get("livro", "Desconhecido"),
        "titulo": chunk.metadata.get("titulo", "Desconhecido"),
        "capitulo": chunk.metadata.get("capitulo", "Desconhecido"),
        "artigo": chunk.metadata.get("artigo", "Desconhecido")
    }

    graph.query("""
        MERGE (livro:Livro {nome: $livro})
        MERGE (titulo:Titulo {nome: $titulo})
        MERGE (capitulo:Capitulo {nome: $capitulo})
        MERGE (artigo:Artigo {nome: $artigo})
        MERGE (chunk:Chunk {id: $chunk_id})
        SET chunk.text = $text

        MERGE (livro)-[:TEM_TITULO]->(titulo)
        MERGE (titulo)-[:TEM_CAPITULO]->(capitulo)
        MERGE (capitulo)-[:TEM_ARTIGO]->(artigo)
        MERGE (artigo)-[:TEM_CHUNK]->(chunk)

        WITH chunk
        CALL db.create.setNodeVectorProperty(chunk, 'textEmbedding', $embedding)
    """, properties)

graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }};
""")

llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="gpt-3.5-turbo"
)

prompt = PromptTemplate.from_template("""
Você é um especialista em extração de grafos semânticos a partir de textos jurídicos em português do Brasil.

Extraia as entidades e relacionamentos presentes no texto abaixo. Identifique os tipos de entidade com rótulos em **português**, como por exemplo: Pessoa, Organização, Local, Documento, Processo, Órgão, etc.

Texto:
{input}

Responda no formato JSON com nós e relacionamentos.
""")

doc_transformer = LLMGraphTransformer(llm=llm, prompt=prompt)

for chunk in all_chunks:
    chunk_id = chunk.metadata["chunk_id"]
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    for graph_doc in graph_docs:
        chunk_node = Node(id=chunk_id, type="Chunk")

        for node in graph_doc.nodes:
            graph_doc.relationships.append(
                Relationship(source=chunk_node, target=node, type="HAS_ENTITY")
            )

        graph.add_graph_documents([graph_doc])
