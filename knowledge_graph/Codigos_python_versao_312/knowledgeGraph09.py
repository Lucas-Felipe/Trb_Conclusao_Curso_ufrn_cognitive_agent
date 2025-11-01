# Versão atualizada do código com melhorias sugeridas:
# - Normalização de entidades
# - Uso de spaCy + EntityRuler para consistência
# - Extração com LLM mais controlada
# - Relacionamentos semânticos mais precisos

import os
import re
import spacy
from spacy.pipeline import EntityRuler
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.prompts import PromptTemplate
from langchain_community.graphs.graph_document import Node, Relationship

load_dotenv()
# ------------------------------------------------------------------------------------------------
# algumas entidades estáticas
# Carregar spaCy e configurar EntityRuler para entidades jurídicas
nlp = spacy.load("pt_core_news_lg")
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [

    # Pessoa
    {"label": "Pessoa", "pattern": [{"LEMMA": {"IN": ["herdeiro", "testador", "cônjuge", "tutor", "curador", "autor", "réu", "sócio"]}}]},
    {"label": "Pessoa", "pattern": [{"LOWER": {"REGEX": "ju[ií]z(es)?"}}]},
    {"label": "Pessoa", "pattern": [{"LOWER": "representante"}, {"LOWER": "legal"}]},
    
    # Instituto
    {"label": "Instituto", "pattern": [{"LOWER": "usucapião"}]},
    {"label": "Instituto", "pattern": [{"LOWER": "testamento"}]},
    {"label": "Instituto", "pattern": [{"LEMMA": "contrato"}]},
    {"label": "Instituto", "pattern": [{"LOWER": "inventário"}]},
    {"label": "Instituto", "pattern": [{"LOWER": "obrigação"}]},
    {"label": "Instituto", "pattern": [{"LOWER": "posse"}]},
    {"label": "Instituto", "pattern": [{"LOWER": "propriedade"}]},
    {"label": "Instituto", "pattern": [{"LOWER": "adoção"}]},
    {"label": "Instituto", "pattern": [{"LOWER": "fiança"}]},
    
    # Objeto Jurídico
    {"label": "Objeto", "pattern": [{"LOWER": "imóvel"}]},
    {"label": "Objeto", "pattern": [{"LOWER": "bem"}, {"LOWER": "móvel"}]},
    {"label": "Objeto", "pattern": [{"LOWER": "empresa"}]},
    {"label": "Objeto", "pattern": [{"LOWER": "coisa"}]},
    {"label": "Objeto", "pattern": [{"LOWER": "produto"}]},
    
    # TempoLegal
    {"label": "TempoLegal", "pattern": [{"LOWER": {"REGEX": "prazo[s]?"}}]},
    {"label": "TempoLegal", "pattern": [{"LOWER": "vencimento"}]},
    {"label": "TempoLegal", "pattern": [{"LOWER": "termo"}]},
    {"label": "TempoLegal", "pattern": [{"LOWER": "data"}, {"LOWER": "limite"}]},
    
    # Órgão
    {"label": "Orgão", "pattern": [{"LOWER": "tribunal"}]},
    {"label": "Orgão", "pattern": [{"LOWER": "ministério"}, {"LOWER": {"REGEX": "p[úu]blico"}}]},
    {"label": "Orgão", "pattern": [{"LOWER": "cartório"}]},
    {"label": "Orgão", "pattern": [{"LOWER": "juizado"}]},
    
    # Documento
    {"label": "Documento", "pattern": [{"LOWER": "certidão"}]},
    {"label": "Documento", "pattern": [{"LOWER": "escritura"}]},
    {"label": "Documento", "pattern": [{"LOWER": "sentença"}]},
    {"label": "Documento", "pattern": [{"LOWER": "contrato"}, {"LOWER": "social"}]},
    {"label": "Documento", "pattern": [{"LOWER": "registro"}, {"LOWER": "civil"}]},
    
    # Norma
    {"label": "Norma", "pattern": [{"LOWER": {"REGEX": "^art(\\.|igo)?$"}}, {"IS_DIGIT": True}]},
    {"label": "Norma", "pattern": [{"LOWER": "parágrafo"}, {"LOWER": "único"}]},
    {"label": "Norma", "pattern": [{"TEXT": {"REGEX": "^§\\s*\\d+º?$"}}]},
    {"label": "Norma", "pattern": [{"LOWER": "inciso"}, {"TEXT": {"REGEX": "^[IVXLCDM]+$"}}]},
    {"label": "Norma", "pattern": [{"LOWER": "lei"}, {"IS_DIGIT": True}]},
    {"label": "Norma", "pattern": [{"LOWER": "código"}, {"LOWER": "civil"}]},
]

ruler.add_patterns(patterns)

normalizacao = {
    # Artigos e elementos da norma
    "artigo": "Artigo",
    "artigo de lei": "Artigo",
    "art.": "Artigo",
    "documento": "Artigo",        # usado quando o modelo confunde o artigo com tipo de doc
    "inciso": "Inciso",
    "parágrafo": "Parágrafo",
    "parágrafo único": "Parágrafo",
    "§": "Parágrafo",

    # Pessoas e papéis
    "juiz": "Pessoa",
    "autor": "Pessoa",
    "réu": "Pessoa",
    "herdeiro": "Pessoa",
    "cônjuge": "Pessoa",
    "tutor": "Pessoa",
    "curador": "Pessoa",
    "representante legal": "Pessoa",
    "sócio": "Pessoa",
    "advogado": "Pessoa",

    # Instituições e órgãos
    "tribunal": "Orgão",
    "cartório": "Orgão",
    "ministério público": "Orgão",
    "juizado": "Orgão",
    "vara": "Orgão",

    # Documentos e instrumentos
    "certidão": "Documento",
    "escritura": "Documento",
    "sentença": "Documento",
    "contrato social": "Documento",
    "registro civil": "Documento",
    "procuração": "Documento",

    # Procedimentos e atos
    "decisão": "Ato",
    "ato": "Ato",
    "ação judicial": "Procedimento",
    "processo": "Procedimento",
    "recurso": "Procedimento",
    "audiencia": "Procedimento",

    # Institutos jurídicos
    "usucapião": "Instituto",
    "testamento": "Instituto",
    "contrato": "Instituto",
    "obrigação": "Instituto",
    "posse": "Instituto",
    "propriedade": "Instituto",
    "adoção": "Instituto",
    "inventário": "Instituto",
    "fiança": "Instituto",
    "penhora": "Instituto",

    # Objetos jurídicos
    "imóvel": "Objeto",
    "bem móvel": "Objeto",
    "empresa": "Objeto",
    "coisa": "Objeto",
    "produto": "Objeto",

    # Tempo legal
    "prazo": "TempoLegal",
    "data": "TempoLegal",
    "termo": "TempoLegal",
    "vencimento": "TempoLegal",
    
    # Norma jurídica
    "lei": "Norma",
    "código civil": "Norma",
}


def normalizar_label(label):
    return normalizacao.get(label.lower(), label.title())
# ----------------------------------------------------------------------------------------------------
# separa o texto em chunks por regex de normas
# Diretório e leitura dos PDFs
DOCS_PATH = "C:/Users/LUCAS FELIPE/Desktop/knowledge_graph/cod_juri"
loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Função de split estruturado

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

        paragrafos = re.split(r"(?=^\s*(\u00a7\s*\d+\u00ba|Parágrafo único))", artigo_completo, flags=re.MULTILINE)
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

all_chunks = []
for doc in docs:
    chunks = split_legal_text(doc.page_content)
    all_chunks.extend(chunks)
# ---------------------------------------------------------------------------------
# cria o grafo e faz as relações
embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    database="codigocivil09"
)

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
# ---------------------------------------------------------------------------------------------
# extração de entidades
# LLM com prompt controlado para enriquecer
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'), 
    model_name="gpt-3.5-turbo"
)

prompt = PromptTemplate.from_template("""
Você é um especialista em Direito e grafos jurídicos.

Extraia os **conceitos jurídicos relevantes** do texto abaixo no seguinte formato JSON:

{
  "nodes": [
    {{"id": "NomeDoConceito", "type": "TipoDoConceito"}}
  ],
  "relationships": [
    {{"source": "ConceitoOrigem", "target": "ConceitoAlvo", "type": "TIPO_RELACAO"}}
  ]
}

- Os tipos de conceito podem incluir: "Instituto", "Objeto", "Pessoa", "Documento", "Orgão", "TempoLegal", etc.
- As relações podem incluir: "REGULA", "ENVOLVE", "MENCIONA", "RESTRINGE", "PERMITE", "PROÍBE", entre outras.
- Seja preciso e evite termos genéricos como "lei", "documento", "artigo".
- Utilize termos jurídicos consolidados e normalize as categorias de forma consistente.

Texto:
{input}
""")


doc_transformer = LLMGraphTransformer(llm=llm, prompt=prompt)

# Extração e inserção dos grafos semânticos com normalização
for chunk in all_chunks:
    chunk_id = chunk.metadata["chunk_id"]
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    for graph_doc in graph_docs:
        chunk_node = Node(id=chunk_id, type="Chunk")

        for node in graph_doc.nodes:
            node.type = normalizar_label(node.type)
            graph_doc.relationships.append(
                Relationship(source=chunk_node, target=node, type="MENCIONA")
            )

        graph.add_graph_documents([graph_doc])
