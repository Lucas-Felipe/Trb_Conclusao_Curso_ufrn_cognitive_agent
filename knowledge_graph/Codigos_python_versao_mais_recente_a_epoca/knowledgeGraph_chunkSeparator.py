# CÓDIGO CONTENDO SEPARAÇÃO DE CHUNKS POR ARTIGO, INCISO, PARÁGRAFO E ETC

import os
import re
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.prompts import PromptTemplate
from langchain_community.graphs.graph_document import Node, Relationship

load_dotenv()

# Carregamento dos documentos PDF
DOCS_PATH = "C:/Users/LUCAS FELIPE/Desktop/knowledge_graph/cod_juri"
loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Função de split jurídico estruturado
def split_legal_text(text):
    artigo_pattern = r"(?=^\s*Art\.?\s*\d+[ºo]?[^\n]*)"
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
        chunk_principal = paragrafos[0].strip()

        documentos = [chunk_principal]

        for i in range(1, len(paragrafos), 2):
            paragrafo = paragrafos[i] + paragrafos[i + 1] if (i + 1) < len(paragrafos) else paragrafos[i]
            documentos.append(paragrafo.strip())

        for doc in documentos:
            incisos = re.split(r"(?=^\s*[IVXLCDM]+\s*-\s+)", doc, flags=re.MULTILINE)
            for inciso in incisos:
                if inciso.strip():
                    chunk_text = inciso.strip()
                    chunk_id = f"{artigo_num}_{hash(chunk_text)}"
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

# Aplicação do split em todos os documentos carregados
all_chunks = []
for doc in docs:
    chunks = split_legal_text(doc.page_content)
    all_chunks.extend(chunks)

# Inicialização dos serviços
embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    database="codigocivil05"
)

# Inserção dos chunks no grafo
for chunk in all_chunks:
    chunk_id = chunk.metadata["chunk_id"]
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    properties = {
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding,
        "livro": chunk.metadata.get("livro", ""),
        "titulo": chunk.metadata.get("titulo", ""),
        "capitulo": chunk.metadata.get("capitulo", ""),
        "artigo": chunk.metadata.get("artigo", "")
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

# Criação do índice vetorial
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }};
""")

# Inicialização do LLM e do extrator de entidades
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

# Extração das entidades com o LLM e mapeamento no grafo
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
