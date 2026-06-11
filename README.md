# 🔍 RAG-based Fake News Detector

**Stack:** `sentence-transformers` · `FAISS` · `Anthropic Claude` · `Streamlit`

A Retrieval-Augmented Generation (RAG) system for fact-checking news claims. Instead of relying purely on LLM memory (which can hallucinate), the system retrieves relevant evidence from a trusted knowledge base before generating a verdict.

## Pipeline

```
User claim
    ↓
Sentence Embedding  (all-MiniLM-L6-v2, 384-dim)
    ↓
FAISS Cosine Search  (IndexFlatIP on normalized vectors)
    ↓
Top-K Evidence Retrieval
    ↓
LLM Prompt Augmentation  (claim + retrieved context)
    ↓
Claude LLM Classification
    ↓
Verdict: REAL / FAKE / UNCERTAIN + explanation
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Enter your Anthropic API key in the sidebar.

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app |
| `RAG_Fake_News_Detector.ipynb` | Jupyter notebook (step-by-step) |
| `requirements.txt` | Python dependencies |

## Architecture Decisions

- **FAISS `IndexFlatIP`**: Exact inner product search. Combined with L2-normalized vectors, this gives cosine similarity. Scales to millions of vectors.
- **`all-MiniLM-L6-v2`**: 6-layer MiniLM, 384-dim embeddings. Fast and accurate for semantic similarity tasks.
- **RAG over pure LLM**: Provides evidence-grounded verdicts and explainability. Knowledge base can be updated without retraining.

## Production Extensions

- Load documents from PDFs/URLs using LangChain document loaders
- Use `RecursiveCharacterTextSplitter` for long document chunking
- Replace FAISS with Pinecone or ChromaDB for persistent storage
- Add a cross-encoder re-ranker for improved retrieval precision
