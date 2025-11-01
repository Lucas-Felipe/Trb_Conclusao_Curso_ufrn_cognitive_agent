import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain.prompts import PromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo",
    temperature=0
)

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="codigocivil07"
)

# Prompt totalmente em português
qa_prompt_template = """
Você é um assistente jurídico especializado nas leis brasileiras.
Use apenas as informações extraídas do contexto da legislação armazenada no grafo para responder à pergunta, de forma clara e objetiva.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""

qa_prompt = PromptTemplate(
    template=qa_prompt_template,
    input_variables=["context", "question"]
)

# Criando retriever baseado em vetores
chunk_vector = Neo4jVector.from_existing_index(
    embedding=embedding,
    graph=graph,
    index_name="chunkVector",
    node_label="Chunk",
    embedding_node_property="textEmbedding",
    text_node_property="text"
)

# Construção da cadeia RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=chunk_vector.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt}
)

# Consulta direta
pergunta = "Do que se trata o art 564 do código civil?"
resposta = qa_chain.invoke(pergunta)

print("🔎 Resposta:")
print(resposta)

# Fecha conexão
graph._driver.close()
