# Benchmark RAG — Boeing 737 AMM + Avaliação RAGAS

Pipeline de **RAG (Retrieval-Augmented Generation)** sobre o manual de manutenção
do Boeing 737 (AMM), com **avaliação automática via RAGAS** usando um juiz LLM
independente (NVIDIA NIM).

## Arquitetura

```
Pergunta ──► 1. TOC (amm_toc_database.json)     ◄── busca semântica TF-IDF
             2. Task identificada no TOC
             3. Arquivo ATA (ex.: 05___084.json)  ◄── restrição por capítulo
             4. Chunks da task/seção              ◄── ranque TF-IDF
             5. Fallback global
                        │
                        ▼
              LLM gerador (NVIDIA NIM) ──► resposta
                        │
                        ▼
              RAGAS: faithfulness, answer_relevancy,
              context_precision, context_recall  ──► CSV + relatório
```

- **Gerador:** `nvidia/nemotron-3-super-120b-a12b`
- **Juiz (avaliador):** `nvidia/nemotron-3.5-super-vl-120b-a12b`
- **Embeddings:** `nvidia/nemotron-3-embed-1b`
- **Retrieval:** TF-IDF em camadas (TOC → arquivo ATA → task), sem uso de
  informação do ground truth na recuperação

## Estrutura

```
src/
  rag_tfidf.py       # pipeline RAG (recuperação + geração)
  evaluate_ragas.py  # avaliação RAGAS + relatório/CSV
  prepare_dataset.py # converte o ground truth para data/train.json
requirements.txt     # dependências com versões fixadas
```

## Como executar

```powershell
# 1. ambiente
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# NOTA: se a instalação do ragas falhar (scikit-network sem wheel para
# Python 3.14), use: pip install --no-deps ragas==0.4.3

# 2. chave da API (arquivo .env na raiz — NÃO versione este arquivo)
#    NVIDIA_API_KEY=nvapi-...   (obtenha em build.nvidia.com)

# 3. pipeline — 5 questões sorteadas (seed aleatória a cada execução)
python src\rag_tfidf.py -r -n 5

# 4. avaliação RAGAS
python src\evaluate_ragas.py
```

Resultados em `output/predictions.json` e `output/resultado_ragas_metrics.csv`.

## Sobre o corpus (não incluído)

A pasta `data/` (corpus do manual AMM Boeing 737, TOC e ground truth com
`source_excerpt`) **não está versionada** por conter material proprietário da
Boeing. Para executar o benchmark, obtenha os JSONs do manual no seu ambiente
de curso e coloque-os em `data/corpus/` — o código os carrega automaticamente.

## Configurações úteis

| Comando | Efeito |
|---|---|
| `-n 5` | limita a 5 questões |
| `-r` | sorteia questões aleatoriamente (seed nova a cada run) |
| `-s 123` | seed fixa (reprodutibilidade) |
