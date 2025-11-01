# ISSO NÃO DEU CERTO
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

# === Inicialização ===

# Conecta ao Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))
db_name = "codigocivil07"

# Carrega o modelo local de QA (em português ou genérico)
qa_pipeline = pipeline("question-answering", model="pierreguillou/bert-base-cased-squad-v1.1-portuguese")

# Carrega modelo de embeddings local
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Função para recuperar chunks mais similares
def recuperar_chunks(pergunta, top_k=5):
    with driver.session(database=db_name) as session:
        result = session.run("MATCH (c:Chunk) RETURN c.id AS id, c.text AS text, c.embedding AS embedding")
        chunks = []
        embeddings = []
        ids = []

        for record in result:
            if record["embedding"]:  # Garante que o embedding está presente
                ids.append(record["id"])
                chunks.append(record["text"])
                embeddings.append(record["embedding"])

    pergunta_embedding = embedding_model.encode([pergunta])
    chunk_embeddings = np.array(embeddings)

    # Similaridade
    sims = cosine_similarity(pergunta_embedding, chunk_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]

    top_chunks = [chunks[i] for i in top_indices]
    return top_chunks

# Faz pergunta e responde
def responder_pergunta(pergunta):
    top_chunks = recuperar_chunks(pergunta)

    melhores_respostas = []
    for chunk in top_chunks:
        try:
            resposta = qa_pipeline(question=pergunta, context=chunk)
            if resposta["score"] > 0.2:  # Filtro de confiança
                melhores_respostas.append((resposta["answer"], resposta["score"]))
        except:
            continue

    if melhores_respostas:
        melhores_respostas.sort(key=lambda x: x[1], reverse=True)
        return melhores_respostas[0][0]
    else:
        return "Não encontrei uma resposta com confiança suficiente nos documentos."

# === Teste ===
pergunta = "Do que se trata o art 1.996 do código civil?"
resposta = responder_pergunta(pergunta)

print("🔎 Resposta:")
print(resposta)

# Fecha conexão
driver.close()
