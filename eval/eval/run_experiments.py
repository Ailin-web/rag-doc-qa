import csv
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

# --- paths: work no matter where the script is started from ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rag"))

from rag.loader import load_pdf
from rag.chunker import chunk_pages
from rag.embed import embed_texts

SIZES = [100, 200, 400]   # chunk size in words (overlap = 20% of size)
KS = [1, 3, 5, 8]         # top-k values to compare


def embed_all(texts):
    key = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    cache_dir = os.path.join(HERE, "results", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, key + ".npy")
    if os.path.exists(cache_file):
        return np.load(cache_file)
    parts = []
    batch = 20
    for i in range(0, len(texts), batch):
        for attempt in range(8):
            try:
                parts.append(embed_texts(texts[i:i + batch]))
                break
            except RuntimeError:
                print(f"  quota hit, waiting 60s (attempt {attempt + 1}/8)...")
                time.sleep(60)
        else:
            raise RuntimeError("Still over quota. Try again later.")
        time.sleep(3)
    out = np.vstack(parts)
    np.save(cache_file, out)
    return out


def normalize(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def is_hit(chunk, source, pages):
    if chunk["source"] != source:
        return False
    return any(chunk["page_start"] <= p <= chunk["page_end"] for p in pages)


with open(os.path.join(HERE, "questions.json"), encoding="utf-8") as f:
    questions = [q for q in json.load(f) if q["source"]]  # answerable only

print(f"Embedding {len(questions)} questions...")
q_vecs = normalize(embed_all([q["question"] for q in questions]))

pdf_paths = sorted(glob.glob(os.path.join("data", "*.pdf")))
print("PDFs:", [os.path.basename(p) for p in pdf_paths])
pages_by_pdf = {os.path.basename(p): load_pdf(p) for p in pdf_paths}

table = []
for size in SIZES:
    overlap = size // 5
    chunks = []
    for name, pages in pages_by_pdf.items():
        chunks += chunk_pages(pages, name, chunk_size=size, overlap=overlap)
    print(f"\nChunk size {size} (overlap {overlap}): {len(chunks)} chunks, embedding...")
    vecs = normalize(embed_all([c["text"] for c in chunks]))
    sims = q_vecs @ vecs.T

    span = sum(c["page_end"] - c["page_start"] + 1 for c in chunks) / len(chunks)
    row = {"chunk_size": size, "overlap": overlap, "num_chunks": len(chunks),
           "avg_pages_per_chunk": round(span, 2)}
    for k in KS:
        hits = 0
        for qi, q in enumerate(questions):
            order = np.argsort(-sims[qi])[:k]
            if any(is_hit(chunks[j], q["source"], q["pages"]) for j in order):
                hits += 1
        row[f"top{k}"] = round(100 * hits / len(questions), 1)
    table.append(row)
    print(row)

# --- save and print the results table ---
out_dir = os.path.join(HERE, "results")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "experiments.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
    w.writeheader()
    w.writerows(table)

print("\n===== EXPERIMENT RESULTS (hit rate %, 27 answerable questions) =====")
print(f"{'chunk':>6} {'chunks':>7} {'pages/chunk':>12} " + " ".join(f"{'top'+str(k):>6}" for k in KS))
for r in table:
    print(f"{r['chunk_size']:>6} {r['num_chunks']:>7} {r['avg_pages_per_chunk']:>12} "
          + " ".join(f"{r['top'+str(k)]:>6}" for k in KS))
print("\nSaved: eval/eval/results/experiments.csv")