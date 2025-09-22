import sqlite3
import numpy as np
import faiss
from utils import load_faiss_index, DB_FILE

index, idmap = load_faiss_index()

def get_chunk_by_id(chunk_id):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, title, url, page, text FROM documents WHERE chunk_id=?", (chunk_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {"chunk_id": row[0], "title": row[1], "url": row[2], "page": row[3], "text": row[4]}
    return None

def baseline_query(query, k=5, embed_fn=None):
    if embed_fn is None:
        from embed_index import embed, tokenizer, model
        def embed_fn(q):
            import torch
            tokens = tokenizer([q], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                out = model(**tokens)
                emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            faiss.normalize_L2(emb)
            return emb

    qvec = embed_fn(query)
    D, I = index.search(qvec, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(idmap):
            chunk_id = idmap[idx]
            chunk = get_chunk_by_id(chunk_id)
            if chunk:
                chunk["vec_score"] = float(score)
                results.append(chunk)
    return results

if __name__ == "__main__":
    print(baseline_query("What are safety precautions?", k=3))
