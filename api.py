# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from reranker import rerank
from baseline_search import baseline_query, get_chunk_by_id
import sqlite3
from utils import set_seed
import uvicorn
set_seed()

app = FastAPI()

class AskRequest(BaseModel):
    q: str
    k: Optional[int] = 5
    mode: Optional[str] = "hybrid"  # "baseline" or "hybrid"
    alpha: Optional[float] = 0.6

@app.post("/ask")
def ask(req: AskRequest):
    q = req.q
    k = req.k or 5
    mode = req.mode
    alpha = req.alpha if req.alpha is not None else 0.6

    if mode == "baseline":
        items = baseline_query(q, k=k)
        reranker_used = "baseline"
    else:
        items = rerank(q, k=k, alpha=alpha)
        reranker_used = f"hybrid(alpha={alpha})"

    # simple decision threshold: if top final_score (or vec_score) below threshold -> abstain
    top_score = items[0]["final_score"] if "final_score" in items[0] else items[0].get("vec_score", 0.0)
    THRESH = 0.22
    if top_score < THRESH:
        return {
            "answer": None,
            "reason": f"Top evidence score {top_score:.3f} < threshold {THRESH}",
            "contexts": items,
            "reranker_used": reranker_used
        }

    # Build extractive answer: take top chunk(s) and return first 1-2 sentences containing query terms if possible
    from nltk.tokenize import sent_tokenize
    def extract_answer_from_chunk(chunk_text):
        sents = sent_tokenize(chunk_text)
        q_terms = set(q.lower().split())
        for s in sents:
            if any(t in s.lower() for t in q_terms):
                return s
        # fallback first sentence
        return sents[0] if sents else chunk_text

    # choose top 1 chunk to form answer (optionally include top2)
    top_chunk = items[0]
    answer_text = extract_answer_from_chunk(top_chunk["text"])
    # include citation
    citation = {
        "chunk_id": top_chunk["chunk_id"],
        "title": top_chunk["title"],
        "url": top_chunk["url"],
        "page": top_chunk["page"]
    }
    return {
        "answer": answer_text,
        "citation": citation,
        "contexts": items,
        "reranker_used": reranker_used
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
