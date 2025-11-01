from fastapi import FastAPI
from pydantic import BaseModel
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

app = FastAPI()

# modelo de embeddings (384 dimensões)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# driver do Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))
banco = "codigocivil10"

class Mensagem(BaseModel):
    mensagem: str
    numero: str

@app.post("/processar")
def processar(m: Mensagem):
    # Gera embedding da pergunta
    embedding = model.encode(m.mensagem).tolist()

    with driver.session(database=banco) as session:
        result = session.run(
            """
            CALL db.index.vector.queryNodes('chunkVector', 3, $vetor)
            YIELD node, score
            RETURN node, score
            """,
            {"vetor": embedding}
        )

        chunks = []
        for record in result:
            node = record["node"]
            # Tenta pegar "texto" (Artigo) ou "text" (Chunk)
            text = node.get("texto") or node.get("text")
            print(f"Debug: node={node}, score={record['score']}, text={text}")  # print de depuração
            if text:
                chunks.append(text)

    resposta = "\n---\n".join(chunks) if chunks else "Não encontrei nada."
    return {"resposta": resposta}
