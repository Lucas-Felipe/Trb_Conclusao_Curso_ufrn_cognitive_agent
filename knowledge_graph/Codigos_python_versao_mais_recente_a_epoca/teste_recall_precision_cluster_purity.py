import os
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

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from collections import Counter, defaultdict
import numpy as np

load_dotenv()

# Configuração
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
    verbose=True,
)

# Pergunta fixa no código
# pergunta = "Do que se trata o artigo 789?"
# response = agent_executor.invoke({"input": pergunta})
# print(response["output"])

# ================== Avaliação ======================
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

test_queries = [
    {
        "question": "O que trata o artigo 789?",
        "relevant_terms": ["artigo 789", "execução", "cumprimento da sentença"]
    },
    {
        "question": "Quais são as regras sobre doação no Código Civil?",
        "relevant_terms": ["doação", "contrato de doação", "revogação da doação"]
    },
]

def precision_recall_at_k(query, relevant_terms, retriever, k=5):
    results = retriever.get_relevant_documents(query)[:k]
    retrieved_texts = [doc.page_content.lower() for doc in results]
    relevant_hits = 0
    for text in retrieved_texts:
        if any(term.lower() in text for term in relevant_terms):
            relevant_hits += 1
    recall = relevant_hits / len(relevant_terms)
    precision = relevant_hits / k
    return precision, recall

for q in test_queries:
    prec, rec = precision_recall_at_k(q["question"], q["relevant_terms"], chunk_vector.as_retriever(), k=5)
    print(f"Pergunta: {q['question']}")
    print(f"Precision@5: {prec:.2f} | Recall@5: {rec:.2f}\n")

# ================== Cluster Purity ======================

def get_all_embeddings_and_labels(graph):
    data = graph.query("""
    MATCH (c:Chunk)
    RETURN c.id AS id, c.text AS text, c.textEmbedding AS embedding
    """)
    embeddings = []
    labels = []
    texts = []
    for row in data:
        if row["embedding"]:
            embeddings.append(row["embedding"])
            texts.append(row["text"])
            text_l = row["text"].lower()
            if "doação" in text_l:
                labels.append("doacao")
            elif "execução" in text_l:
                labels.append("execucao")
            elif "posse" in text_l:
                labels.append("posse")
            else:
                labels.append("outro")
    return np.array(embeddings), labels, texts

embeddings, true_labels, texts = get_all_embeddings_and_labels(graph)
k = len(set(true_labels))
kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
pred_labels = kmeans.labels_

cluster_to_labels = defaultdict(list)
for pred, true in zip(pred_labels, true_labels):
    cluster_to_labels[pred].append(true)

majority_labels = {
    cluster: Counter(labels).most_common(1)[0][0] for cluster, labels in cluster_to_labels.items()
}

correct = sum(
    1 for pred, true in zip(pred_labels, true_labels) if majority_labels[pred] == true
)
purity = correct / len(true_labels)
print(f"Cluster purity: {purity:.2f}")

graph._driver.close()
