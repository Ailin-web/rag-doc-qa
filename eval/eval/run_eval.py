import csv
import json
import os
import sys
import time

# --- paths: work no matter where the script is started from ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rag"))

from rag.store import load_index, search
from rag.generate import answer

PAUSE = 4  # seconds between questions, to be gentle on the free API quota


def is_hit(chunk, source, pages):
    """Hit = chunk is from the right PDF and covers at least one correct page."""
    if chunk["source"] != source:
        return False
    return any(chunk["page_start"] <= p <= chunk["page_end"] for p in pages)


def is_refusal(text):
    return "find that in the documents" in text.lower()


def pct(a, b):
    return f"{a}/{b} = {100 * a / b:.0f}%" if b else "n/a"


chunks, vectors = load_index()
with open(os.path.join(HERE, "questions.json"), encoding="utf-8") as f:
    questions = json.load(f)

rows = []
for i, q in enumerate(questions, 1):
    row = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "reference_answer": q["answer"],
        "gold_source": q["source"],
        "gold_pages": q["pages"],
        "hit3": None,
        "hit5": None,
        "generated_answer": "",
        "sources": "",
        "refused": None,
        "error": "",
        "my_mark": "",
    }
    try:
        results = search(q["question"], chunks, vectors, top_k=5)
        if q["source"]:
            row["hit3"] = any(is_hit(r, q["source"], q["pages"]) for r in results[:3])
            row["hit5"] = any(is_hit(r, q["source"], q["pages"]) for r in results[:5])
        text, sources = answer(q["question"], chunks, vectors, top_k=4)
        row["generated_answer"] = text
        row["sources"] = ", ".join(sources)
        row["refused"] = is_refusal(text)
    except Exception as e:
        row["error"] = str(e)
    rows.append(row)
    print(f"[{i}/{len(questions)}] {q['id']} {q['type']:<12} "
          f"hit3={row['hit3']} hit5={row['hit5']} refused={row['refused']}"
          + (f" ERROR: {row['error'][:60]}" if row["error"] else ""))
    time.sleep(PAUSE)

# --- save results ---
out_dir = os.path.join(HERE, "results")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)
with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# --- summary ---
answerable = [r for r in rows if r["gold_source"] and not r["error"]]
unans = [r for r in rows if not r["gold_source"] and not r["error"]]
print("\n===== RESULTS =====")
print("Errors (not scored):", sum(1 for r in rows if r["error"]))
print("Retrieval Top-3 hit rate:", pct(sum(r["hit3"] for r in answerable), len(answerable)))
print("Retrieval Top-5 hit rate:", pct(sum(r["hit5"] for r in answerable), len(answerable)))
for t in ["direct", "reworded", "textbook"]:
    sub = [r for r in answerable if r["type"] == t]
    print(f"  {t:<9} Top-3: {pct(sum(r['hit3'] for r in sub), len(sub))}   "
          f"Top-5: {pct(sum(r['hit5'] for r in sub), len(sub))}")
print("Correct refusals (unanswerable):", pct(sum(r["refused"] for r in unans), len(unans)))
print("False refusals (answerable but refused):",
      pct(sum(r["refused"] for r in answerable), len(answerable)))
print("\nSaved: eval/eval/results/results.csv and results.json")