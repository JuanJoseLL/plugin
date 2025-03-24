import faiss
import numpy as np
import os

INDEX_PATH = "data/faiss_index/global.index"
CHUNKS_PATH = "data/faiss_index/chunks.txt"

def create_faiss_index(embeddings, chunks):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    vectors = np.array(embeddings).astype("float32")
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.replace("\n", " ") + "\n")
