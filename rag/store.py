import json
import os

import numpy as np

from chunker import chunk_pages
from embed import embed_texts
from loader import load_pdf

INDEX_DIR = "index"


def build_index(pdf_path):
    """Load PDF, chunk it, embed chunks, save everything to the index folder."""
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages)
    vectors = embed_texts([c["text"] for c in chunks])

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(os.path.join(INDEX_DIR, "vectors.npy"), vectors)
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    return chunks, vectors


def load_index():
    """Load the saved chunks and vectors."""
    vectors = np.load(os.path.join(INDEX_DIR, "vectors.npy"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks, vectors


def search(question, chunks, vectors, top_k=3):
    """Return the top_k chunks closest in meaning to the question."""
    q = embed_texts([question])[0]
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    best = np.argsort(scores)[::-1][:top_k]
    return [{"score": float(scores[i]), **chunks[i]} for i in best]


if __name__ == "__main__":
    import sys

    chunks, vectors = build_index(sys.argv[1])
    print("Indexed", len(chunks), "chunks")
    for q in ["What is a kernel?", "What do operating systems do?"]:
        print("\nQuestion:", q)
        for r in search(q, chunks, vectors):
            print(f"  page {r['page']} | score {r['score']:.3f} | {r['text'][:80]}")