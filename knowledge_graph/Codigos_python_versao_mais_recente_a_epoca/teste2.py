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

load_dotenv()

# Configuração
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="codigocivil05"
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

# Vetor com base nos nomes corretos
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
pergunta = "Do que se trata o artigo 789 do código civil?"

response = agent_executor.invoke({"input": pergunta})
print(response["output"])

graph._driver.close()