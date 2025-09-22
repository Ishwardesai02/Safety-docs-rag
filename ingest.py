import os
import sqlite3
from PyPDF2 import PdfReader
from tqdm import tqdm
import json

DB_FILE = "db.sqlite"
DATA_DIR = "D:/Campus X Langchain/RAG/Data/industrial-safety-pdfs"

def create_table(conn):
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS documents")
    c.execute("""
        CREATE TABLE documents (
            chunk_id TEXT PRIMARY KEY,
            title TEXT,
            url TEXT,
            page INTEGER,
            text TEXT
        )
    """)
    conn.commit()

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ingest_pdfs():
    conn = sqlite3.connect(DB_FILE)
    create_table(conn)
    cur = conn.cursor()

    with open(os.path.join(DATA_DIR, "D:/Campus X Langchain/RAG/Data/sources.json"), "r") as f:
        sources = json.load(f)

    for src in sources:
        title, url, fname = src["title"], src["url"], src["filename"]
        pdf_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(pdf_path):
            print(f"[WARN] Missing {pdf_path}")
            continue

        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            chunks = chunk_text(text)
            for i, ch in enumerate(chunks):
                chunk_id = f"{fname}_p{page_num}_c{i}"
                cur.execute("INSERT INTO documents VALUES (?,?,?,?,?)",
                            (chunk_id, title, url, page_num, ch))

    conn.commit()
    conn.close()
    print(f"[INFO] Ingestion complete -> {DB_FILE}")

if __name__ == "__main__":
    ingest_pdfs()
