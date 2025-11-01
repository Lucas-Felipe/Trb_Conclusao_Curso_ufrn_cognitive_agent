# Esse código cria o grafo utilizando sentence-transformers para embeddings
# space para identificar entidades listadas
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

# === Função para dividir o texto em chunks estruturados por elementos jurídicos ===
def dividir_em_chunks_estruturados(texto):
    padrao = r"(?=Art\.?\s*\d+º?.*|§\s*\d+º?.*|[IVXLCDM]+\s*-\s.*|\(\w\)\s)"
    partes = re.split(padrao, texto)
    return [p.strip() for p in partes if len(p.strip()) > 50]

# === Função para extrair contexto legal (Parte, Livro, Título, Capítulo, Seção) ===
def extrair_contexto(texto):
    contexto = {
        "parte": None,
        "livro": None,
        "titulo": None,
        "capitulo": None,
        "secao": None
    }

    linhas = texto.split("\n")
    for linha in linhas:
        linha = linha.strip()

        if re.match(r"^P\s*A\s*R\s*T\s*E\s+.*", linha, re.IGNORECASE):
            contexto["parte"] = linha.strip()
        elif re.match(r"^LIVRO\s+[IVXLCDM]+", linha, re.IGNORECASE):
            contexto["livro"] = linha.strip()
        elif re.match(r"^T[IÍ]TULO\s+[IVXLCDM]+", linha, re.IGNORECASE):
            contexto["titulo"] = linha.strip()
        elif re.match(r"^CAP[IÍ]TULO\s+[IVXLCDM]+", linha, re.IGNORECASE):
            contexto["capitulo"] = linha.strip()
        elif re.match(r"^SE[CÇ][AÃ]O\s+[IVXLCDM]+", linha, re.IGNORECASE):
            contexto["secao"] = linha.strip()

    return contexto

# === Carregamento do PDF ===
pdf_path = "C:/Users/LUCAS FELIPE/Desktop/knowledge_graph/cod_juri/codigo_civil.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Texto completo do PDF
texto_completo = "\n".join(page.page_content for page in pages)

# Chunking por estrutura jurídica
chunks_texto = dividir_em_chunks_estruturados(texto_completo)

# Nome do banco
nome_do_banco = "codigocivil08"

# === Inserção no Neo4j ===
with driver.session(database=nome_do_banco) as session:
    contexto_atual = {
        "parte": None,
        "livro": None,
        "titulo": None,
        "capitulo": None,
        "secao": None
    }

    for i, texto_chunk in enumerate(chunks_texto):
        # Atualiza contexto se encontrar novas marcações
        novo_contexto = extrair_contexto(texto_chunk)
        for chave in contexto_atual:
            if novo_contexto[chave]:
                contexto_atual[chave] = novo_contexto[chave]

        # Embedding local
        embedding = embedding_provider.encode(texto_chunk).tolist()

        # ID único
        chunk_id = str(uuid.uuid4())

        # Extrai entidades com spaCy
        doc = nlp(texto_chunk)
        entidades_extraidas = list(set(ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2))

        # Extrai artigo (opcional)
        artigo_match = re.search(r"Art\.?\s*(\d+º?)", texto_chunk)
        artigo = f"Artigo {artigo_match.group(1)}" if artigo_match else "Artigo Desconhecido"

        # Inserção Cypher
        session.execute_write(
            run_query,
            """
            MERGE (p:Parte {nome: $parte})
            MERGE (l:Livro {nome: $livro})
            MERGE (t:Titulo {nome: $titulo})
            MERGE (c:Capitulo {nome: $capitulo})
            MERGE (s:Secao {nome: $secao})
            MERGE (a:Artigo {nome: $artigo})
            MERGE (chunk:Chunk {id: $chunk_id})
            SET chunk.text = $texto,
                chunk.embedding = $embedding

            MERGE (p)-[:TEM_LIVRO]->(l)
            MERGE (l)-[:TEM_TITULO]->(t)
            MERGE (t)-[:TEM_CAPITULO]->(c)
            MERGE (c)-[:TEM_SECAO]->(s)
            MERGE (s)-[:TEM_ARTIGO]->(a)
            MERGE (a)-[:TEM_CHUNK]->(chunk)
            """,
            {
                "parte": contexto_atual["parte"] or "Parte Desconhecida",
                "livro": contexto_atual["livro"] or "Livro Desconhecido",
                "titulo": contexto_atual["titulo"] or "Título Desconhecido",
                "capitulo": contexto_atual["capitulo"] or "Capítulo Desconhecido",
                "secao": contexto_atual["secao"] or "Seção Desconhecida",
                "artigo": artigo,
                "chunk_id": chunk_id,
                "texto": texto_chunk,
                "embedding": embedding
            }
        )

        # Relaciona entidades extraídas
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

print("✅ Grafo criado com estrutura completa: Parte, Livro, Título, Capítulo, Seção, Artigo e Chunk.")
