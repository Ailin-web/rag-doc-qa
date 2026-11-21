import sys

sys.path.append("rag")

from store import load_index, search

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

for question, correct_page in QUESTIONS:
    results = search(question, chunks, vectors, top_k=3)
    pages = [r["page"] for r in results]
    ok = correct_page in pages
    hits += ok
    print(("OK   " if ok else "MISS ") + question)
    print("     expected page", correct_page, "| got pages", pages)

print(f"\nRetrieval hit rate: {hits}/{len(QUESTIONS)}")