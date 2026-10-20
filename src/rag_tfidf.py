import os
import json
import random
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Carregar variáveis de ambiente (.env)
load_dotenv(override=True)  # .env do projeto prevalece sobre variaveis de ambiente

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "data" / "corpus"
TRAIN_FILE = BASE_DIR / "data" / "train.json"
OUTPUT_PRED_JSON = BASE_DIR / "output" / "predictions.json"

TOC_FILE = CORPUS_DIR / "amm_toc_database.json"
if not TOC_FILE.exists():
    TOC_FILE = BASE_DIR / "data" / "amm_toc_database.json"


def carregar_e_indexar_toc():
    """Carrega o TOC para a Etapa 1 da busca."""
    if not TOC_FILE.exists():
        return None, None, None

    with open(TOC_FILE, "r", encoding="utf-8") as f:
        toc_data = json.load(f)

    toc_texts = [
        f"{item.get('task_name', '')} {item.get('subject_name', '')} {item.get('chapter_section_subject', '')} ATA {item.get('ata_principal', '')} Task {item.get('task_number', '')}"
        for item in toc_data
    ]

    vectorizer_toc = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_toc_matrix = vectorizer_toc.fit_transform(toc_texts)

    return toc_data, vectorizer_toc, tfidf_toc_matrix


def executar_pipeline_rag(limit: int = None, is_random: bool = False, seed: int = None):
    os.makedirs(OUTPUT_PRED_JSON.parent, exist_ok=True)

    print("1. Indexando Tabela de Conteúdos (TOC)...")
    toc_data, vectorizer_toc, tfidf_toc_matrix = carregar_e_indexar_toc()

    print("2. Carregando seções do manual Boeing 737 em data/corpus/ (JSONs pré-extraídos)...")
    docs = []
    for json_path in sorted(CORPUS_DIR.glob("*.json")):
        if json_path.name == TOC_FILE.name:
            continue  # amm_toc_database.json é usado apenas na Etapa 1 da busca
        with open(json_path, "r", encoding="utf-8") as f:
            secoes = json.load(f)
        if not isinstance(secoes, list):
            continue
        for item in secoes:
            conteudo = item.get("content", "")
            if not conteudo:
                continue
            docs.append(Document(
                page_content=f"{item.get('title', '')}\n{conteudo}",
                metadata={
                    "source": json_path.name,
                    "task_id": item.get("task_id", ""),
                    "chapter": item.get("chapter", ""),
                }
            ))

    if not docs:
        raise FileNotFoundError(
            f"Nenhuma seção encontrada em {CORPUS_DIR}. "
            "Verifique se os JSONs do manual estão nessa pasta."
        )
    print(f"   {len(docs)} seções carregadas do corpus.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    chunk_texts = [c.page_content for c in chunks]

    # Indexação global de backup
    vectorizer_global = TfidfVectorizer(stop_words='english')
    tfidf_global_matrix = vectorizer_global.fit_transform(chunk_texts)

    # ---- Base de conhecimento estruturada ----
    # Codigo de task do manual: NN-NN-NN-NNN-NNN (ex.: 05-00-52-211-805)
    TASK_CODE_RE = re.compile(r"\b(\d{2}-\d{2}-\d{2}-\d{3}-\d{3})\b")
    # Capitulo ATA citado na pergunta: "capitulo 10", "chapter 26", "ATA 32"
    CAPITULO_RE = re.compile(
        r"cap[íi]tulo\s+0*(\d{1,2})\b|chapter\s+0*(\d{1,2})\b|\bATA\s+0*(\d{1,2})\b",
        re.IGNORECASE,
    )

    def normalizar_task_code(valor):
        """Extrai o codigo de task de qualquer string ('TASK 05-...' -> '05-...')."""
        if not valor:
            return None
        m = TASK_CODE_RE.search(str(valor).upper().replace("TASK", " "))
        return m.group(1) if m else None

    def normalizar_capitulo(pergunta):
        """Extrai o capitulo ATA citado na propria pergunta ('capitulo 10' -> '10')."""
        m = CAPITULO_RE.search(str(pergunta))
        if not m:
            return None
        return next(g for g in m.groups() if g).zfill(2)

    # Camada 3 (task): task_code -> chunks daquela task
    # Camada 2 (arquivo ATA): nome do arquivo -> [(task_code, chunk), ...]
    chunks_por_task = {}
    chunks_por_arquivo = {}
    for c in chunks:
        task_code = normalizar_task_code(c.metadata.get("task_id", ""))
        texto = c.page_content
        if task_code:
            chunks_por_task.setdefault(task_code, []).append(texto)
        filename = Path(c.metadata.get("source", "")).name
        chunks_por_arquivo.setdefault(filename, []).append((task_code, texto))

    # TOC indexado por task_number para lookup estruturado
    toc_por_numero = {}
    if toc_data:
        for t in toc_data:
            num = t.get("task_number")
            if num:
                toc_por_numero[num] = t

    def ranquear_pool(pergunta, pool, top_k):
        """Ordena os chunks do pool restrito por relevancia TF-IDF."""
        if not pool:
            return []
        try:
            vectorizer_pool = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            matriz_pool = vectorizer_pool.fit_transform(pool)
            query_vec = vectorizer_pool.transform([pergunta])
            similaridades = cosine_similarity(query_vec, matriz_pool).flatten()
        except ValueError:
            return pool[:top_k]
        k_actual = min(top_k, len(pool))
        top_indices = similaridades.argsort()[-k_actual:][::-1]
        return [pool[i] for i in top_indices]

    def pools_do_ata(ata, secao=""):
        """Camada 2: pools de chunks restritos ao capitulo ATA (arquivo).
        Retorna (pool_secao, pool_arquivo)."""
        pool_arquivo, pool_secao = [], []
        if not ata:
            return pool_secao, pool_arquivo
        for f in chunks_por_arquivo:
            if not f.startswith(ata):
                continue
            for task_code, texto in chunks_por_arquivo[f]:
                pool_arquivo.append(texto)
                if secao and task_code and task_code.startswith(secao):
                    pool_secao.append(texto)
        return pool_secao, pool_arquivo

    def recuperar_contextos(pergunta, top_k=3):
        """
        Base de conhecimento estruturada — busca em camadas, guiada exclusivamente
        pela pergunta (nenhuma informacao do ground truth e usada):

        1. Capitulo: se a pergunta cita o capitulo ATA ("capitulo 10"), restringe
           os candidatos do TOC a esse capitulo.
        2. TOC (amm_toc_database.json): busca semantica TF-IDF pelo texto da
           pergunta -> task/secao/ATA alvo (top-3 candidatos).
        3. Task: restringe aos chunks da task identificada no TOC.
        4. Arquivo/secao: restringe ao arquivo ATA correspondente e, dentro dele,
           a secao identificada — a camada de busca limitada ao que a pergunta pede.
        5. Fallback: TF-IDF global sobre todo o corpus.
        """
        capitulo = normalizar_capitulo(pergunta)

        # Camadas 1 e 2: TOC — top-3 candidatos (priorizando o capitulo citado)
        if toc_data and vectorizer_toc and tfidf_toc_matrix is not None:
            query_toc_vec = vectorizer_toc.transform([pergunta])
            sims_toc = cosine_similarity(query_toc_vec, tfidf_toc_matrix).flatten()

            if capitulo:
                idx_capitulo = [
                    i for i, t in enumerate(toc_data)
                    if str(t.get("ata_principal", "")) == capitulo
                ]
                candidatos = sorted(idx_capitulo, key=lambda i: sims_toc[i], reverse=True)[:3]
            if not capitulo or not candidatos:
                candidatos = [int(i) for i in sims_toc.argsort()[::-1][:3]]

            for idx in candidatos:
                toc_item = toc_data[idx]

                # Camada 3: task exata do candidato
                code = toc_item.get("task_number")
                if code and code in chunks_por_task:
                    return ranquear_pool(pergunta, chunks_por_task[code], top_k)

                # Camada 4: arquivo ATA (+ secao) do candidato
                pool_secao, pool_arquivo = pools_do_ata(
                    str(toc_item.get("ata_principal", "")),
                    str(toc_item.get("chapter_section_subject", "")),
                )
                pool = pool_secao or pool_arquivo
                if pool:
                    return ranquear_pool(pergunta, pool, top_k)

        # Camada 4 (reserva): arquivo do capitulo citado, sem match no TOC
        if capitulo:
            pool_secao, pool_arquivo = pools_do_ata(capitulo)
            pool = pool_secao or pool_arquivo
            if pool:
                return ranquear_pool(pergunta, pool, top_k)

        # Camada 5: fallback global (matriz pre-computada)
        query_vec = vectorizer_global.transform([pergunta])
        similaridades = cosine_similarity(query_vec, tfidf_global_matrix).flatten()
        top_indices = similaridades.argsort()[-top_k:][::-1]
        return [chunk_texts[i] for i in top_indices]

    # Model LLM (meta/llama-3.1-70b-instruct foi descontinuado pela NVIDIA em 2026-08-26)
    llm = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b", temperature=0.0)

    prompt_template = PromptTemplate.from_template("""
Você é um especialista técnico de manutenção aeronáutica do Boeing 737 (AMM).
Com base estritamente no contexto recuperado dos manuais abaixo, responda à pergunta do mecânico de forma precisa e direta. Se a informação não estiver contida no contexto, declare que não possui informações suficientes no manual fornecido.

Contexto Técnico Recuperado:
{context}

Pergunta: {question}

Resposta Técnica:
""")

    # Filtragem das Perguntas
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        perguntas_gt = json.load(f)

    if is_random and limit:
        if seed is None:
            seed = random.randrange(1, 1_000_000)  # sorteio realmente variavel
        random.seed(seed)
        qtd = min(limit, len(perguntas_gt))
        perguntas_gt = random.sample(perguntas_gt, qtd)
        print(f"🎲 Selecionadas {qtd} questões aleatórias (Seed {seed}).")
    elif limit:
        perguntas_gt = perguntas_gt[:limit]
        print(f"⚡ Selecionadas as primeiras {len(perguntas_gt)} questões.")

    predictions = []

    print("\n" + "=" * 80)
    print(" EXECUTANDO GERADOR RAG (TOC + CHUNKS + LLM)")
    print("=" * 80)

    for i, item in enumerate(perguntas_gt, 1):
        q_id = item.get("question_id")
        pergunta = item.get("question")
        gabarito = item.get("ground_truth")

        contextos = recuperar_contextos(pergunta, top_k=3)
        contexto_unificado = "\n\n--- CHUNK ---\n\n".join(contextos)

        prompt_formatado = prompt_template.format(context=contexto_unificado, question=pergunta)
        resposta_llm = llm.invoke(prompt_formatado).content

        print(f"\n[Questão {i}/{len(perguntas_gt)}] ID: {q_id}")
        print(f"❓ PERGUNTA: {pergunta}")
        print(f"📘 GROUND TRUTH: {gabarito}")
        print(f"🤖 RESPOSTA GERADA: {resposta_llm}")
        print("-" * 80)

        # Formato limpo e compatível com RAGAS
        predictions.append({
            "question_id": q_id,
            "question": pergunta,
            "answer": resposta_llm,
            "contexts": contextos,
            "ground_truth": gabarito
        })

    with open(OUTPUT_PRED_JSON, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Execução concluída. Respostas salvas em: {OUTPUT_PRED_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o Pipeline RAG e gera predictions.json")
    parser.add_argument("-n", "--limit", type=int, default=None, help="Quantidade de perguntas")
    parser.add_argument("-r", "--random", action="store_true", help="Amostragem aleatória")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Semente do sorteio (padrao: aleatoria)")

    args = parser.parse_args()

    executar_pipeline_rag(
        limit=args.limit,
        is_random=args.random,
        seed=args.seed
    )
