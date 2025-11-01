import os
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'), 
    temperature=0
)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    database="neo4j"
)

CYPHER_GENERATION_TEMPLATE = """Tarefa:Gerar consultas cypher para usar no banco de dados gráfico.
Instruções:
Use apenas relações, tipos e propriedades vindos do schema.
Não use qualquer outra relação, tipo ou propriedade que não venha do schema.
Apenas inclua a consulta cypher na sua resposta gerada.

Sempre use case sensitive nas buscas por string.

Schema:
{schema}

The question is:
{question}"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

schema_str = graph.get_schema
if len(schema_str) > 5000:
    schema_str = schema_str[:5000] + "\n... (schema truncado)"

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt.partial(schema=schema_str),
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True
)

def run_cypher(q):
    return cypher_chain.invoke({"query": q})

while (q := input("> ")) != "exit":
    print(run_cypher(q))