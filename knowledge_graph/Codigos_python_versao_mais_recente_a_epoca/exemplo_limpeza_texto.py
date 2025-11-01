import re
import pdfplumber
import os

# ——— Funções de limpeza e normalização ———
def limpar_texto(texto):
    texto = ''.join(c for c in texto if c.isprintable() or c in '\n\r\t§ºª\u201c\u201d\u2013\u2014')
    texto = re.sub(r"https?://\S+", "", texto)
    texto = re.sub(r"^\s*\.", "", texto)
    texto = re.sub(r"\b\d{2}/\d{2}/\d{4}, \d{2}:\d{2} L\d+\b", "", texto)
    texto = re.sub(r"\b\d+/\d+\b", "", texto)
    linhas = texto.splitlines()
    linhas_limpas = [re.sub(r'\s+', ' ', linha).strip() for linha in linhas if linha.strip()]
    return "\n".join(linhas_limpas).strip()

def remove_parte_final(texto):
    padrao = r"Brasília,\s+\d{1,2} de [A-Za-z]+ de \d{4}.*?FERNANDO HENRIQUE CARDOSO"
    match = re.search(padrao, texto, flags=re.DOTALL | re.IGNORECASE)
    return texto[:match.start()].strip() if match else texto

# ——— Leitura e processamento do PDF ———
def processar_pdf(caminho_pdf):
    with pdfplumber.open(caminho_pdf) as pdf:
        textos = []
        for i, pagina in enumerate(pdf.pages):
            texto = pagina.extract_text()
            if not texto:
                continue
            clean = limpar_texto(texto)
            clean = remove_parte_final(clean)
            textos.append((i + 1, texto, clean))
        return textos

# ——— Execução principal ———
if __name__ == "__main__":
    caminho_pdf = "C:/Users/LUCAS FELIPE/Desktop/knowledge_graph/cod_juri/codigo_civil.pdf"  # coloque o nome do seu PDF
    resultados = processar_pdf(caminho_pdf)

    if not resultados:
        print("⚠️ Nenhum texto foi extraído do PDF.")
    else:
        # Mostra exemplos no terminal
        for num_pagina, original, limpo in resultados[:2]:
            print("="*100)
            print(f"📄 Página {num_pagina} — Texto Original:")
            print(original[:800], "...\n")
            print("🧹 Texto Após Limpeza e Normalização:")
            print(limpo[:800], "...")
            print("="*100, "\n")

        # Cria nomes dos arquivos
        nome_base = os.path.splitext(caminho_pdf)[0]
        arquivo_original = f"{nome_base}_original.txt"
        arquivo_limpo = f"{nome_base}_limpo.txt"

        # Salva o texto original e o limpo
        with open(arquivo_original, "w", encoding="utf-8") as f_ori, \
             open(arquivo_limpo, "w", encoding="utf-8") as f_limpo:
            for _, original, limpo in resultados:
                f_ori.write(original + "\n\n")
                f_limpo.write(limpo + "\n\n")

        print(f"✅ Texto original salvo em: {arquivo_original}")
        print(f"✅ Texto limpo salvo em: {arquivo_limpo}")


