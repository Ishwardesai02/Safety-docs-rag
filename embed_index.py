import sqlite3
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from utils import save_faiss_index, DB_FILE

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed(texts, batch_size=16):
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(**tokens)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
        vecs.append(emb)
    return np.vstack(vecs)

def build_index():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, text FROM documents")
    rows = cur.fetchall()
    conn.close()

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    embeddings = embed(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    save_faiss_index(index, ids)

if __name__ == "__main__":
    build_index()
