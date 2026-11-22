import glob
import json
import os

import numpy as np

from chunker import chunk_pages
from embed import embed_texts
from loader import load_pdf

INDEX_DIR = "index"
BATCH = 50


def build_index(data_dir="data"):
    """Read every PDF in data_dir, chunk, embed, and save to the index folder."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        name = os.path.basename(path)
        try:
            pages = load_pdf(path)
        except ValueError as e:
            print("Skipping", name, "-", e)
            continue
        new = chunk_pages(pages, name)
        print(name, "->", len(new), "chunks")
        chunks.extend(new)

    texts = [c["text"] for c in chunks]
    vectors = []
    for i in range(0, len(texts), BATCH):
        vectors.append(embed_texts(texts[i:i + BATCH]))
        print("Embedded", min(i + BATCH, len(texts)), "/", len(texts))
    vectors = np.vstack(vectors)

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
    chunks, vectors = build_index()
    print("Indexed", len(chunks), "chunks")
    for q in ["What is a kernel?", "What do operating systems do?"]:
        print("\nQuestion:", q)
        for r in search(q, chunks, vectors):
            print(f"  {r['source']} p{r['page_start']}-{r['page_end']} | {r['score']:.3f} | {r['text'][:70]}")