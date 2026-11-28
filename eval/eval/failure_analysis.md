# Failure Analysis

Setup: 200-word chunks (40 overlap), Top-5 retrieval, Gemini embeddings, gemini-3.5-flash-lite for answers. Test set: 31 questions (14 direct, 5 reworded, 8 textbook, 4 unanswerable).

## Final results

| Measure | Result |
|---|---|
| Retrieval hit rate, Top-3 | 25/27 = 93% |
| Retrieval hit rate, Top-5 | 27/27 = 100% |
| Answer accuracy (hand-marked) | 26/27 = 96% |
| Correct refusals on unanswerable questions | 4/4 = 100% |
| False refusals | 1/27 = 4% |

## Prompt fix (v1 to v2)

| Question | v1 prompt | v2 prompt |
|---|---|---|
| q09 (what multiprocessors share) | Partial: left out the clock | Correct |
| q18 (laptop vs smartphone view) | Wrong: refused | Wrong: refused |

Accuracy went from 25/27 (93%) to 26/27 (96%). The v2 prompt asks for every relevant detail and allows answers from related information.

## Failure categories

1. **Small model too literal (q18).** Retrieval found the right slide, but gemini-3.5-flash-lite refused because the slide lists the differences and never states a "why". The larger gemini-3.8-flash answered it correctly in a single test. It could not be used for the full evaluation because its free quota returned 429 and 503 errors, so the deployed app uses the small model and q18 is a known limitation.
2. **Incomplete answers (q09, fixed).** The detail was in the retrieved text but the model dropped it. Fixed by asking for every relevant detail in the prompt.
3. **Same topic in several PDFs (q02, q23).** The slides and the textbook cover the same material, so a page from another PDF ranked above my chosen page at Top-3. Both were found at Top-5 and answered correctly. This is partly a limit of the test, which counts only one page as correct.
4. **Citation noise.** The Sources list always shows the 4 retrieved chunks, even when only one was used.
5. **Generous page-range hits.** Slides hold few words per page, so even 100-word chunks span 2.3 pages on average, and 400-word chunks span 6.6. Top-1 is the fairest metric.
6. **Free-tier API limits.** The stronger model hit rate limits and busy-server errors, and fallback between models can mix models in one evaluation run.

## Chunk size and Top-K experiment (retrieval hit rate, 27 questions)

| Chunk size | Chunks | Pages per chunk | Top-1 | Top-3 | Top-5 | Top-8 |
|---|---|---|---|---|---|---|
| 100 | 113 | 2.34 | 81.5% | 92.6% | 96.3% | 100% |
| 200 | 57 | 3.74 | 70.4% | 92.6% | 100% | 100% |
| 400 | 29 | 6.62 | 70.4% | 88.9% | 100% | 100% |

Chosen: 200-word chunks with Top-5. One question is 3.7%, so small gaps are not significant.