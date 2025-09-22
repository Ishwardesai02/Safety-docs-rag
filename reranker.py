from rank_bm25 import BM25Okapi
import sqlite3
from baseline_search import baseline_query, get_chunk_by_id
from utils import DB_FILE

def build_bm25():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, text FROM documents")
    rows = cur.fetchall()
    conn.close()

    corpus = [r[1].split() for r in rows]
    idmap = [r[0] for r in rows]
    return BM25Okapi(corpus), idmap

bm25, bm25_ids = build_bm25()

def rerank(query, k=5, alpha=0.6):
    bm25_scores = bm25.get_scores(query.split())
    bm25_dict = {cid: s for cid, s in zip(bm25_ids, bm25_scores)}

    vec_results = baseline_query(query, k=20)
    for r in vec_results:
        bm25_score = bm25_dict.get(r["chunk_id"], 0.0)
        r["bm25_score"] = bm25_score
        r["final_score"] = alpha * r["vec_score"] + (1 - alpha) * bm25_score

    reranked = sorted(vec_results, key=lambda x: -x["final_score"])
    return reranked[:k]

if __name__ == "__main__":
    print(rerank("What are safety precautions?", k=3))
