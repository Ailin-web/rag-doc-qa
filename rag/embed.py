import os
import time

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "gemini-embedding-001"


def embed_texts(texts, retries=3):
    """Turn a list of texts into a numpy array of vectors."""
    for attempt in range(retries):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
            )
            return np.array([e.values for e in result.embeddings])
        except Exception as e:
            print("Embedding failed:", str(e)[:100])
            time.sleep(5)
    raise RuntimeError("Embedding failed after several tries.")


if __name__ == "__main__":
    vectors = embed_texts(["What is an operating system?"])
    print("Vector shape:", vectors.shape)
    print("First 5 numbers:", vectors[0][:5])