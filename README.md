# Mini RAG + Reranker

This project implements a **mini Retrieval-Augmented Generation (RAG)** pipeline with PDF ingestion, embeddings, FAISS vector search, BM25, and a hybrid reranker. It also includes both a **FastAPI backend** and a **Streamlit frontend** for asking questions against ingested documents.




---
   
##  Repository Structure
         mini-rag-reranker/
         ├─ data/
         │ ├─ industrial-safety-pdfs.zip # (your PDF dataset)
         │ ├─ sources.json # list of PDF sources (title, url/path)
         │ └─ questions.txt # evaluation questions
         ├─ ingest.py # ingest + chunk -> sqlite
         ├─ embed_index.py # embeddings + FAISS index
         ├─ baseline_search.py # baseline cosine similarity search
         ├   ─ reranker.py # hybrid reranker (FAISS + BM25)
         ├─ api.py # FastAPI server with POST /ask
         ├─ streamlit_app.py # Streamlit frontend for Q&A
         ├─ utils.py # helper functions
         └─ README.md

## 🚀 Setup

1. Clone the repo and navigate inside:

   ```bash
   git clone https://github.com/your-username/mini-rag-reranker.git
   cd mini-rag-reranker
---
## Usage
1. Ingest PDFs

Parses sources.json, downloads/loads PDFs, chunks them, and saves to db.sqlite.
   
     python ingest.py

2. Build embeddings + FAISS index

Creates vector embeddings and saves FAISS index to disk.

    python embed_index.py


3. Test baseline search

Runs a simple cosine similarity search over embeddings.

    python baseline_search.py


4. Test reranker

Combines FAISS vector search with BM25 to rerank answers.

    python reranker.py


5. Run API (FastAPI)

Start the backend:

    uvicorn api:app --reload


Open API docs at: http://127.0.0.1:8000/docs



Example CURL Requests

Easy query:

    curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is FAISS used for?"}'


Tricky query:

    curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "How does reranking improve retrieval over pure FAISS search?"}'

## Features
- **Document Ingestion:** Parses and indexes PDF documents from the `Data/industrial-safety-pdfs` and `pdfs/` directories.
- **Embedding Index:** Uses FAISS for fast similarity search over document embeddings.
- **API Server:** Exposes endpoints for querying the document collection.
- **Reranking:** Improves search results using a reranker module.
- **Baseline Search:** Provides a baseline search method for comparison.
- **SQLite Database:** Stores metadata and indexing information.






