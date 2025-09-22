# RAG (Retrieval-Augmented Generation) Project

This project implements a Retrieval-Augmented Generation (RAG) pipeline for document search and question answering over a collection of industrial safety documents. It leverages embedding-based search, FAISS indexing, and reranking to provide relevant answers from a curated set of PDFs and other resources.

## Features
- **Document Ingestion:** Parses and indexes PDF documents from the `Data/industrial-safety-pdfs` and `pdfs/` directories.
- **Embedding Index:** Uses FAISS for fast similarity search over document embeddings.
- **API Server:** Exposes endpoints for querying the document collection.
- **Reranking:** Improves search results using a reranker module.
- **Baseline Search:** Provides a baseline search method for comparison.
- **SQLite Database:** Stores metadata and indexing information.

## Project Structure
```
api.py                # API server (FastAPI/Flask)
baseline_search.py    # Baseline search implementation
db.sqlite             # SQLite database for metadata
embed_index.py        # Embedding and FAISS index logic
faiss.index           # FAISS index file
idmap.pkl             # Mapping of document IDs
ingest.py             # Document ingestion and parsing
reranker.py           # Reranking logic
utils.py              # Utility functions
Data/                 # Source PDFs and metadata
pdfs/                 # Additional PDF resources
```

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation
1. Clone the repository or copy the project files.
2. (Optional) Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install required packages:
   ```powershell
   pip install -r requirements.txt
   ```
   *(Create `requirements.txt` if not present, with packages like `faiss-cpu`, `numpy`, `pandas`, `fastapi`, `uvicorn`, `PyPDF2`, etc.)*

### Usage
- **Ingest Documents:**
  ```powershell
  python ingest.py
  ```
- **Build Embedding Index:**
  ```powershell
  python embed_index.py
  ```
- **Run API Server:**
  ```powershell
  python api.py
  ```
  The server will start and listen for requests (default: http://127.0.0.1:8000/).

### Querying
Use the API endpoint to submit queries and receive relevant document passages.

## Data
- Place your PDF files in `Data/industrial-safety-pdfs/` or `pdfs/`.
- Metadata and sources are tracked in `Data/sources.json`.

## License
Specify your license here (e.g., MIT, Apache 2.0).

## Acknowledgements
- Inspired by retrieval-augmented generation research and open-source projects.
- Uses FAISS for vector search and FastAPI/Flask for the API layer.
