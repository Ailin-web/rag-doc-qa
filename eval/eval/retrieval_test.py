import sys

sys.path.append("rag")

from store import load_index, search

SOURCE = "Unit-1.pdf"

# (question, correct page in Unit-1.pdf)
QUESTIONS = [
    ("What is a kernel?", 7),
    ("What is the goal of an operating system?", 3),
    ("What are the two viewpoints of what an OS does?", 6),
    ("What is stored in ROM that loads the OS when the computer powers on?", 8),
    ("What are the levels of the storage hierarchy and which are volatile?", 11),
    ("How does a device controller tell the CPU an I/O operation has finished?", 13),
    ("What is the difference between maskable and non-maskable interrupts?", 18),
    ("How does the CPU find the right handler when an interrupt occurs?", 17),
    ("What is the difference between asymmetric and symmetric multiprocessing?", 21),
    ("What is a clustered system and what does it provide?", 24),
]

chunks, vectors = load_index()
hits = 0
hits3 = 0

for question, page in QUESTIONS:
    results = search(question, chunks, vectors, top_k=5)
    ok = any(
        r["source"] == SOURCE and r["page_start"] <= page <= r["page_end"]
        for r in results
    )
    hits += ok
    hits3 += any(r["source"] == SOURCE and r["page_start"] <= page <= r["page_end"] for r in results[:3])
    print(("OK   " if ok else "MISS ") + question)
    print("     expected", SOURCE, "p" + str(page), "| got",
          [f"{r['source'][:12]} p{r['page_start']}-{r['page_end']}" for r in results])

print(f"\nTop 3: {hits3}/{len(QUESTIONS)} | Top 5: {hits}/{len(QUESTIONS)}")