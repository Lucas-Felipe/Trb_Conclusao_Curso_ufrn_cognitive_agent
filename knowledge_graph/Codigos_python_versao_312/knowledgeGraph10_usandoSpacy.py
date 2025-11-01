# === Importações ===
import re
import spacy
from neo4j import GraphDatabase
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import uuid

# === Inicialização ===
nlp = spacy.load("pt_core_news_sm")
embedding_provider = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(username, password))
nome_do_banco = "codigocivil10"

def limpar_grafo(tx):
    tx.run("MATCH (n) DETACH DELETE n")

with driver.session(database=nome_do_banco) as session:
    session.execute_write(limpar_grafo)

def run_query(tx, query, parameters=None):
    tx.run(query, parameters or {})

def dividir_em_chunks_estruturados(texto):
    padrao = r"(?=Art\.?\s*\d+(?:º|\.)?)"  # pega Art. 1º, Art. 1., Art. 1
    matches = list(re.finditer(padrao, texto))
    chunks = []
    
    # Verifica se há conteúdo antes do primeiro match
    if matches and matches[0].start() > 0:
        primeiro_trecho = texto[:matches[0].start()].strip()
        if len(primeiro_trecho) > 50:
            chunks.append(primeiro_trecho)

    for i, match in enumerate(matches):
        inicio = match.start()
        fim = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
        trecho = texto[inicio:fim].strip()
        if len(trecho) > 50:
            chunks.append(trecho)
    return chunks


def extrair_contexto_com_nomes(texto):
    contexto = {
        "parte": (None, None),
        "livro": (None, None),
        "titulo": (None, None),
        "capitulo": (None, None),
        "secao": (None, None)
    }
    linhas = texto.split("\n")
    i = 0
    while i < len(linhas):
        linha = linhas[i].strip()
        if re.match(r"^P\s*A\s*R\s*T\s*E\s+.*", linha, re.IGNORECASE):
            nome = linhas[i+1].strip() if i+1 < len(linhas) else ""
            contexto["parte"] = (linha, nome)
        elif re.match(r"^LIVRO\s+[IVXLCDM]+", linha, re.IGNORECASE):
            nome = linhas[i+1].strip() if i+1 < len(linhas) else ""
            contexto["livro"] = (linha, nome)
        elif re.match(r"^T[IÍ]TULO\s+[IVXLCDM]+", linha, re.IGNORECASE):
            nome = linhas[i+1].strip() if i+1 < len(linhas) else ""
            contexto["titulo"] = (linha, nome)
        elif re.match(r"^CAP[IÍ]TULO\s+[IVXLCDM]+", linha, re.IGNORECASE):
            nome = linhas[i+1].strip() if i+1 < len(linhas) else ""
            contexto["capitulo"] = (linha, nome)
        elif re.match(r"^SE[CÇ][AÃ]O\s+[IVXLCDM]+", linha, re.IGNORECASE):
            nome = linhas[i+1].strip() if i+1 < len(linhas) else ""
            contexto["secao"] = (linha, nome)
        i += 1
    return contexto

def limpar_texto(texto):
    texto = ''.join(c for c in texto if c.isprintable() or c in '\n\r\t§ºª\u201c\u201d\u2013\u2014')
    texto = re.sub(r"https?://\S+", "", texto)
    texto = re.sub(r"^\s*\.", "", texto)
    texto = re.sub(r"\b\d{2}/\d{2}/\d{4}, \d{2}:\d{2} L10406\b", "", texto)
    texto = re.sub(r"\b\d+/\d+\b", "", texto)
    linhas = texto.splitlines()
    linhas_limpas = [re.sub(r'\s+', ' ', linha).strip() for linha in linhas if linha.strip() != '']
    return "\n".join(linhas_limpas).strip()

def extrair_incisos(texto):
    padrao = r"(?:^|\n)[ \t]*([IVXLCDM]+)[\.\-\–\—]?[ \t]+(.*?)(?=(?:\n[ \t]*[IVXLCDM]+[\.\-\–\—]?[ \t]+)|\n[a-z]\)|\n§|\nArt\.|\Z)"
    return re.findall(padrao, texto, re.DOTALL)

def extrair_alineas(texto):
    padrao = r"(?:\n|\r|\n\r)[ \t]*([a-z])\)[ \t]+(.*?)(?=(?:\n[ \t]*[a-z]\)\s)|\n[A-Z]|$)"
    return re.findall(padrao, texto, re.DOTALL)

def remove_parte_final(texto):
    # Padrão que captura o início da parte final
    padrao = r"Brasília,\s+\d{1,2} de janeiro de 2002.*?FERNANDO HENRIQUE CARDOSO.*?(ÍNDICE|P\s*A\s*R\s*T\s*E)"
    match = re.search(padrao, texto, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return texto[:match.start()].strip()
    return texto

# === Carregamento do PDF ===
pdf_path = "C:/Users/LUCAS FELIPE/Desktop/knowledge_graph/cod_juri/codigo_civil.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

paginas_tratadas = []
for page in pages:
    linhas = page.page_content.splitlines()
    paginas_tratadas.append("\n".join(linhas))

texto_completo = "\n".join(paginas_tratadas)
texto_completo = limpar_texto(texto_completo)
texto_completo = remove_parte_final(texto_completo)
chunks_texto = dividir_em_chunks_estruturados(texto_completo)

with driver.session(database=nome_do_banco) as session:
    contexto_atual = {
        "parte": (None, None),
        "livro": (None, None),
        "titulo": (None, None),
        "capitulo": (None, None),
        "secao": (None, None)
    }

    for texto_chunk in chunks_texto:
        novo_contexto = extrair_contexto_com_nomes(texto_chunk)
        for chave in contexto_atual:
            if novo_contexto[chave][0]:
                contexto_atual[chave] = novo_contexto[chave]

        embedding = embedding_provider.encode(texto_chunk).tolist()
        chunk_id = str(uuid.uuid4())
        doc = nlp(texto_chunk)
        entidades_extraidas = list(set(ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2))

        artigo_match = re.search(r"Art\.?\s*(\d+(?:\.\d+)*)(-[A-Z])?(?:º)?", texto_chunk, re.IGNORECASE)
        if artigo_match:
            numero = artigo_match.group(1)
            sufixo = artigo_match.group(2) or ""
            artigo = f"Artigo {numero}{sufixo}"
        else:
            artigo = "Artigo Desconhecido"

        texto_sem_contexto = []
        for linha in texto_chunk.splitlines():
            if not re.match(r"^(CAP[IÍ]TULO|T[IÍ]TULO|LIVRO|PARTE|SE[CÇ][AÃ]O)\s+[IVXLCDM]+", linha, re.IGNORECASE):
                texto_sem_contexto.append(linha.strip())
        texto_chunk_limpo = "\n".join(texto_sem_contexto).strip()

        texto_artigo_principal = re.split(r"§\s*\d+º?", texto_chunk_limpo)[0].strip()
        texto_artigo_sem_titulo = re.sub(r"^Art\.?\s*\d+(?:º|\.)?\s*-?\s*", "", texto_artigo_principal, flags=re.IGNORECASE).strip()

        session.execute_write(run_query, """
            MERGE (p:Parte {nome: $parte})
            SET p.titulo = $parte_nome
            MERGE (l:Livro {nome: $livro})
            SET l.titulo = $livro_nome
            MERGE (t:Titulo {nome: $titulo})
            SET t.titulo = $titulo_nome
            MERGE (c:Capitulo {nome: $capitulo})
            SET c.titulo = $capitulo_nome
            MERGE (s:Secao {nome: $secao})
            SET s.titulo = $secao_nome
            MERGE (a:Artigo {nome: $artigo})
            SET a.texto = $texto_artigo
            MERGE (chunk:Chunk {id: $chunk_id})
            SET chunk.text = $texto,
                chunk.embedding = $embedding
            MERGE (p)-[:TEM_LIVRO]->(l)
            MERGE (l)-[:TEM_TITULO]->(t)
            MERGE (t)-[:TEM_CAPITULO]->(c)
            MERGE (c)-[:TEM_SECAO]->(s)
            MERGE (s)-[:TEM_ARTIGO]->(a)
            MERGE (a)-[:TEM_CHUNK]->(chunk)
        """, {
            "parte": contexto_atual["parte"][0] or "Parte Desconhecida",
            "parte_nome": contexto_atual["parte"][1] or "",
            "livro": contexto_atual["livro"][0] or "Livro Desconhecido",
            "livro_nome": contexto_atual["livro"][1] or "",
            "titulo": contexto_atual["titulo"][0] or "Título Desconhecido",
            "titulo_nome": contexto_atual["titulo"][1] or "",
            "capitulo": contexto_atual["capitulo"][0] or "Capítulo Desconhecido",
            "capitulo_nome": contexto_atual["capitulo"][1] or "",
            "secao": contexto_atual["secao"][0] or "Seção Desconhecida",
            "secao_nome": contexto_atual["secao"][1] or "",
            "artigo": artigo,
            "texto_artigo": texto_artigo_sem_titulo,
            "chunk_id": chunk_id,
            "texto": texto_chunk_limpo,
            "embedding": embedding
        })

        for entidade in entidades_extraidas:
            session.execute_write(run_query, """
                MERGE (e:Entidade {nome: $nome})
                MERGE (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(e)
            """, {"nome": entidade, "chunk_id": chunk_id})

        padrao_paragrafo = re.compile(
            r'^§\s*(\d+[ºo]?)\s+(.*?)(?=^§\s*\d+[ºo]?|^Art\.|^CAPÍTULO|^TÍTULO|\Z)',
            re.MULTILINE | re.DOTALL
        )
        paragrafos = padrao_paragrafo.findall(texto_chunk_limpo)  # lista de tuplas (numero, texto)
        paragrafos_extraidos = [f"§ {numero} {texto.strip()}" for numero, texto in paragrafos]

        texto_restante = texto_chunk_limpo
        for p in paragrafos_extraidos:
            texto_restante = texto_restante.replace(p, "")

        for paragrafo_texto in paragrafos_extraidos:
            paragrafo_id = str(uuid.uuid4())
            session.execute_write(run_query, """
                MATCH (a:Artigo {nome: $artigo})
                MERGE (p:Paragrafo {id: $pid})
                SET p.texto = $texto
                MERGE (a)-[:TEM_PARAGRAFO]->(p)
            """, {"artigo": artigo, "pid": paragrafo_id, "texto": paragrafo_texto.strip()})

            for inciso_num, inciso_texto in extrair_incisos(paragrafo_texto):
                inciso_id = str(uuid.uuid4())
                session.execute_write(run_query, """
                    MATCH (p:Paragrafo {id: $pid})
                    MERGE (i:Inciso {id: $iid})
                    SET i.numero = $numero, i.texto = $texto
                    MERGE (p)-[:TEM_INCISO]->(i)
                """, {
                    "pid": paragrafo_id,
                    "iid": inciso_id,
                    "numero": inciso_num.strip(),
                    "texto": inciso_texto.strip()
                })

                for letra, texto_alinea in extrair_alineas(inciso_texto):
                    alinea_id = str(uuid.uuid4())
                    session.execute_write(run_query, """
                        MATCH (i:Inciso {id: $iid})
                        MERGE (al:Alinea {id: $aid})
                        SET al.letra = $letra, al.texto = $texto
                        MERGE (i)-[:TEM_ALINEA]->(al)
                    """, {
                        "iid": inciso_id,
                        "aid": alinea_id,
                        "letra": letra.strip(),
                        "texto": texto_alinea.strip()
                    })

        for inciso_num, inciso_texto in extrair_incisos(texto_restante):
            inciso_id = str(uuid.uuid4())
            session.execute_write(run_query, """
                MATCH (a:Artigo {nome: $artigo})
                MERGE (i:Inciso {id: $iid})
                SET i.numero = $numero, i.texto = $texto
                MERGE (a)-[:TEM_INCISO]->(i)
            """, {
                "artigo": artigo,
                "iid": inciso_id,
                "numero": inciso_num.strip(),
                "texto": inciso_texto.strip()
            })

            for letra, texto_alinea in extrair_alineas(inciso_texto):
                alinea_id = str(uuid.uuid4())
                session.execute_write(run_query, """
                    MATCH (i:Inciso {id: $iid})
                    MERGE (al:Alinea {id: $aid})
                    SET al.letra = $letra, al.texto = $texto
                    MERGE (i)-[:TEM_ALINEA]->(al)
                """, {
                    "iid": inciso_id,
                    "aid": alinea_id,
                    "letra": letra.strip(),
                    "texto": texto_alinea.strip()
                })

print("\u2705 Grafo completo com Partes, Livros, Títulos, Capítulos, Seções, Artigos, Parágrafos, Incisos e Alíneas.")
