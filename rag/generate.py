import os
import time

from dotenv import load_dotenv
from google import genai

from store import load_index, search

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# If a model is retired or busy, the next one is tried. Change names here only.
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash"]

PROMPT = """You answer questions using ONLY the context below.
If the answer is not in the context, reply exactly: I couldn't find that in the documents.
Keep the answer short and clear. Do not use outside knowledge.

Context:
{context}

Question: {question}
Answer:"""


def ask_llm(prompt):
    for name in MODELS:
        for attempt in range(2):
            try:
                return client.models.generate_content(model=name, contents=prompt).text
            except Exception as e:
                print("Model", name, "failed:", str(e)[:80])
                time.sleep(3)
    raise RuntimeError("All models failed. Try again in a minute.")


def answer(question, chunks, vectors, top_k=4):
    results = search(question, chunks, vectors, top_k=top_k)
    context = "\n\n".join(
        f"[{r['source']} pages {r['page_start']}-{r['page_end']}]\n{r['text']}"
        for r in results
    )
    text = ask_llm(PROMPT.format(context=context, question=question))
    sources = [f"{r['source']} p{r['page_start']}-{r['page_end']}" for r in results]
    return text, sources


if __name__ == "__main__":
    import sys

    chunks, vectors = load_index()
    q = " ".join(sys.argv[1:]) or "What is a kernel?"
    text, sources = answer(q, chunks, vectors)
    print("\nAnswer:", text)
    print("\nSources:")
    for s in sources:
        print(" -", s)