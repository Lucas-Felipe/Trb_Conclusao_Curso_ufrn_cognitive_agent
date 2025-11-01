# ESSE CÓDIGO USA DA LIMPEZA E DO STOP WORDS
import os
import re
import unicodedata
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.prompts import PromptTemplate
from langchain_community.graphs.graph_document import Node, Relationship

load_dotenv()

import re
import unicodedata
import nltk
from nltk.corpus import stopwords

# Baixar os stopwords do NLTK, se necessário
nltk.download('stopwords')
stop_words_pt = set(stopwords.words('portuguese'))

# Adicione quaisquer outras stopwords personalizadas aqui, se quiser
both_stopwords = stop_words_pt

def remove_stopwords(text):
    tokens = text.split()
    tokens = filter(lambda token: token not in both_stopwords, tokens)
    return " ".join(tokens)

def limpar_texto(texto: str) -> str:
    linhas = texto.splitlines()
    linhas_limpa = []

    for linha in linhas:
        if re.match(r'^\d{2}/\d{2}/\d{4}, \d{2}:\d{2} L\d+', linha):
            continue
        if re.search(r'https?://\S+|www\.\S+', linha):
            continue
        if re.match(r'^\s*[A-ZÁÉÍÓÚÂÊÔÃÕÇa-z].*\.+\s*\d+\s*$', linha):
            continue
        if re.match(r'^\s*\d+\s*$', linha):
            continue
        if re.search(r'\bíndice\b', linha, re.IGNORECASE):
            continue
        linha = unicodedata.normalize('NFKD', linha).encode('ASCII', 'ignore').decode('utf-8', 'ignore')
        linhas_limpa.append(linha.strip())

    texto_limpo = ' '.join(linhas_limpa)
    texto_limpo = texto_limpo.lower()
    texto_limpo = re.sub(r'[^\w\s]', '', texto_limpo)
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    texto_limpo = remove_stopwords(texto_limpo)  

    return texto_limpo



DOCS_PATH = "C:/Users/LUCAS FELIPE/Desktop/knowledge_graph/cod_juri"
loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(docs)

chunks_limpos = []
for chunk in chunks:
    texto_limpo = limpar_texto(chunk.page_content)
    novo_chunk = Document(
        page_content=texto_limpo,
        metadata=chunk.metadata
    )
    chunks_limpos.append(novo_chunk)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    database="codigocivil03"
)

for chunk in chunks_limpos:
    filename = os.path.basename(chunk.metadata["source"])
    chunk_id = f"{filename}.{chunk.metadata['page']}"
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }
    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
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

doc_transformer = LLMGraphTransformer(
    llm=llm,
    prompt=prompt
)

for chunk in chunks_limpos:
    filename = os.path.basename(chunk.metadata["source"])
    chunk_id = f"{filename}.{chunk.metadata['page']}"
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )
        for node in graph_doc.nodes:
            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node, 
                    type="HAS_ENTITY"
                )
            )
    graph.add_graph_documents(graph_docs)
