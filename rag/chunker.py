def clean_text(text):
    """Remove odd symbols (like broken bullets) but keep normal text."""
    return "".join(ch if (ch.isprintable() or ch.isspace()) else " " for ch in text)


def chunk_pages(pages, source, chunk_size=200, overlap=40):
    """Cut one document into overlapping chunks.

    Each chunk records the file name and the first and last page it covers.
    """
    words = []
    word_pages = []
    for page in pages:
        text = clean_text(page["text"])
        for w in text.split():
            words.append(w)
            word_pages.append(page["page"])

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append({
            "source": source,
            "page_start": word_pages[start],
            "page_end": word_pages[end - 1],
            "text": " ".join(words[start:end]),
        })
        if end >= len(words):
            break
        start = end - overlap
    return chunks


if __name__ == "__main__":
    import os
    import sys
    from loader import load_pdf

    pages = load_pdf(sys.argv[1])
    chunks = chunk_pages(pages, os.path.basename(sys.argv[1]))
    print("Total chunks:", len(chunks))
    for c in chunks[:3]:
        print("---", c["source"], "| pages", c["page_start"], "-", c["page_end"])
        print(c["text"][:150])