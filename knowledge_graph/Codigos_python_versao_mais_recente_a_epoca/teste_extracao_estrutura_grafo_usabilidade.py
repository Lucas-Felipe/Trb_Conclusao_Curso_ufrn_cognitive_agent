import os
import time
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import Tool
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from sklearn.metrics import precision_score, recall_score, f1_score

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="codigocivil04"
)

qa_prompt_template = """
Você é um assistente jurídico especializado nas leis brasileiras.
Utilize as informações da legislação do grafo para responder a pergunta de forma clara e em português.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""

qa_prompt = PromptTemplate(
    template=qa_prompt_template,
    input_variables=["context", "question"],
)

law_chat = qa_prompt | llm | StrOutputParser()

chunk_vector = Neo4jVector.from_existing_index(
    embedding=embedding,
    graph=graph,
    index_name="chunkVector",
    node_label="Chunk",
    embedding_node_property="textEmbedding",
    text_node_property="text"
)

law_retriever = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=chunk_vector.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt},
)

tools = [
    Tool.from_function(
        name="Consulta Legislação",
        description="Use para responder perguntas com base no texto da legislação armazenada. A entrada será uma pergunta em linguagem natural.",
        func=law_retriever.invoke,
    ),
    Tool.from_function(
        name="Chat Jurídico",
        description="Use para conversas sobre temas jurídicos em geral. A entrada será uma pergunta em linguagem natural.",
        func=law_chat.invoke,
    ),
]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

# Avaliação de métricas
queries = [
    {
        "pergunta": "O que trata o artigo 789?",
        "relevantes": ["art. 789", "responsabilidade da execução"]
    },
    {
        "pergunta": "Quais são as regras sobre doação no Código Civil?",
        "relevantes": ["doação", "contrato de doação", "regras de doação"]
    }
]

retriever = chunk_vector.as_retriever()
k = 5
all_precisions, all_recalls, all_f1s = [], [], []

print("\n🎯 Avaliação de Recuperação:")
for q in queries:
    pergunta = q["pergunta"]
    gabarito = q["relevantes"]

    retrieved_docs = retriever.get_relevant_documents(pergunta)[:k]
    retrieved_texts = [doc.page_content.lower() for doc in retrieved_docs]

    y_true = [any(term.lower() in doc for doc in retrieved_texts) for term in gabarito]
    y_pred = [True] * len(gabarito)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nPergunta: {pergunta}")
    print(f"Precision@{k}: {precision:.2f} | Recall@{k}: {recall:.2f} | F1: {f1:.2f}")

    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(f1)

# 📊 Métricas do grafo
print("\n📊 Avaliação da Estrutura do Grafo:")
try:
    cobertura_query = """
        MATCH (c:Chunk)
        OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e)
        WITH c, count(e) AS entidades
        RETURN count(c) AS total, count(CASE WHEN entidades > 0 THEN 1 END) AS com_entidade
    """
    cobertura = pd.DataFrame(graph.query(cobertura_query))
    total = cobertura["total"].iloc[0]
    com_entidade = cobertura["com_entidade"].iloc[0]
    print(f"Cobertura de Chunks (com pelo menos uma entidade): {com_entidade}/{total} = {(com_entidade/total):.2f}")

    grau_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS grau
        RETURN avg(grau) AS grau_medio
    """
    grau_medio = pd.DataFrame(graph.query(grau_query))["grau_medio"].iloc[0]
    print(f"Grau médio dos nós: {grau_medio:.2f}")

    densidade_query = """
        MATCH (n)
        WITH count(n) AS N
        MATCH ()-[r]-()
        WITH N, count(r) AS E
        RETURN (2.0 * E) / (N * (N - 1)) AS densidade
    """
    densidade = pd.DataFrame(graph.query(densidade_query))["densidade"].iloc[0]
    print(f"Densidade do grafo: {densidade:.4f}")

except Exception as e:
    print(f"Erro ao consultar estrutura do grafo: {e}")

# ⏱️ Tempo de inferência
print("\n⏱️ Tempo de inferência para consulta:")
start = time.time()
resposta = agent_executor.invoke({"input": "Do que se trata a união estável no Código Civil?"})
end = time.time()
print(f"Tempo: {end - start:.2f} segundos")
print("Resposta:", resposta["output"])

graph._driver.close()
