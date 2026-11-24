import os
import tempfile

import numpy as np

from chunker import chunk_pages
from embed import embed_texts
from loader import load_pdf

BATCH = 50
MAX_MB = 20


def index_uploaded_pdf(file_bytes, name):
    """Chunk and embed an uploaded PDF in memory. Returns (chunks, vectors)."""
    if len(file_bytes) > MAX_MB * 1024 * 1024:
        raise ValueError(f"File is too large (limit {MAX_MB} MB).")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(file_bytes)
        path = f.name
    try:
        pages = load_pdf(path)  # raises ValueError for scanned PDFs
    finally:
        os.remove(path)

    chunks = chunk_pages(pages, name)
    parts = []
    for i in range(0, len(chunks), BATCH):
        batch = [c["text"] for c in chunks[i:i + BATCH]]
        parts.append(embed_texts(batch))
    return chunks, np.vstack(parts)