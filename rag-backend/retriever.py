import faiss
import numpy as np

INDEX_PATH = "data/faiss_index/global.index"
CHUNKS_PATH = "data/faiss_index/chunks.txt"

def search_similar_chunks(query_embedding, k=3):
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = f.readlines()

    D, I = index.search(np.array([query_embedding]).astype("float32"), k)
    return [chunks[i].strip() for i in I[0] if i < len(chunks)]
