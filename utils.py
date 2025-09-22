import os
import random
import numpy as np
import torch
import faiss
import pickle

INDEX_FILE = "faiss.index"
IDMAP_FILE = "idmap.pkl"
DB_FILE = "db.sqlite"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_faiss_index(index, idmap):
    faiss.write_index(index, INDEX_FILE)
    with open(IDMAP_FILE, "wb") as f:
        pickle.dump(idmap, f)
    print(f"[INFO] Saved index -> {INDEX_FILE}, idmap -> {IDMAP_FILE}")

def load_faiss_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(IDMAP_FILE):
        raise FileNotFoundError("Run embed_index.py first to create FAISS index + idmap.pkl")
    index = faiss.read_index(INDEX_FILE)
    with open(IDMAP_FILE, "rb") as f:
        idmap = pickle.load(f)
    return index, idmap
