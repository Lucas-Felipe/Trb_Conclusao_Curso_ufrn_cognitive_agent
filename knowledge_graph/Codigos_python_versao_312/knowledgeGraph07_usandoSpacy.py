import os
import re
import spacy
from neo4j import GraphDatabase
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import uuid

# === Inicialização ===

# Carrega modelo spaCy para português
nlp = spacy.load("pt_core_news_sm")

# Modelo de embeddings local
embedding_provider = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Conexão com o Neo4j
uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(username, password))

# Função para executar queries Cypher
def run_query(tx, query, parameters=None):
    tx.run(query, parameters or {})

# === Função para dividir texto em chunks estruturados pela lei ===
def dividir_em_chunks_estruturados(texto):
    # Regex para separar por Artigos, Parágrafos, Incisos e Alíneas
    # Explicação:
    #   - (?=...) é lookahead para dividir sem perder o padrão
    #   - Detecta "Art. 1º", "§ 1º", "I -", "(a)" etc
    padrao = r"(?=Art\.?\s*\d+º?.*|§\s*\d+º?.*|[IVXLCDM]+\s*-\s.*|\(\w\)\s)"
    partes = re.split(padrao, texto)
    partes_filtradas = [p.strip() for p in partes if len(p.strip()) > 50]  # Filtra partes muito pequenas
    return partes_filtradas

# === Carregamento do PDF e extração do texto completo ===

pdf_path = "C:/Users/LUCAS FELIPE/Desktop/knowledge_graph/cod_juri/codigo_civil.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Concatena todo o texto do PDF em uma string só para chunking estruturado
texto_completo = "\n".join(page.page_content for page in pages)

# Aplica chunking estruturado
chunks_texto = dividir_em_chunks_estruturados(texto_completo)

# Banco Neo4j
nome_do_banco = "codigocivil07"

# === Inserção no Neo4j ===

with driver.session(database=nome_do_banco) as session:
    for i, texto_chunk in enumerate(chunks_texto):
        # Embedding local
        embedding = embedding_provider.encode(texto_chunk).tolist()

        # ID único para o chunk
        chunk_id = str(uuid.uuid4())

        # Extrai entidades com spaCy
        doc = nlp(texto_chunk)
        entidades_extraidas = list(set(ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2))

        # Extração simples da estrutura — pode melhorar com regex se quiser extrair o número do artigo real
        artigo_match = re.search(r"Art\.?\s*(\d+º?)", texto_chunk)
        artigo = f"Artigo {artigo_match.group(1)}" if artigo_match else "Artigo Desconhecido"

        livro = "Livro I"      # Você pode melhorar para extrair do texto se disponível
        titulo = "Título I"    # Idem
        capitulo = "Capítulo I" # Idem

        # Inserção Cypher
        session.execute_write(
            run_query,
            """
            MERGE (livro:Livro {nome: $livro})
            MERGE (titulo:Titulo {nome: $titulo})
            MERGE (capitulo:Capitulo {nome: $capitulo})
            MERGE (artigo:Artigo {nome: $artigo})
            MERGE (chunk:Chunk {id: $chunk_id})
            SET chunk.text = $texto,
                chunk.embedding = $embedding

            MERGE (livro)-[:TEM_TITULO]->(titulo)
            MERGE (titulo)-[:TEM_CAPITULO]->(capitulo)
            MERGE (capitulo)-[:TEM_ARTIGO]->(artigo)
            MERGE (artigo)-[:TEM_CHUNK]->(chunk)
            """,
            {
                "livro": livro,
                "titulo": titulo,
                "capitulo": capitulo,
                "artigo": artigo,
                "chunk_id": chunk_id,
                "texto": texto_chunk,
                "embedding": embedding
            }
        )

        # Entidades
        for entidade in entidades_extraidas:
            session.execute_write(
                run_query,
                """
                MERGE (e:Entidade {nome: $nome})
                MERGE (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(e)
                """,
                {"nome": entidade, "chunk_id": chunk_id}
            )

print("✅ Processo concluído com chunking estruturado por Artigos, Parágrafos e Incisos!")
