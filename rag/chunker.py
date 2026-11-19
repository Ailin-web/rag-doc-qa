def clean_text(text):
    """Remove odd symbols (like broken bullets) but keep normal text."""
    return "".join(ch if (ch.isprintable() or ch.isspace()) else " " for ch in text)


def chunk_pages(pages, chunk_size=200, overlap=40):
    """Join all pages into one word stream, then cut into overlapping chunks.

    Each chunk remembers the page where it starts.
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
        end = start + chunk_size
        chunks.append({
            "page": word_pages[start],
            "text": " ".join(words[start:end]),
        })
        if end >= len(words):
            break
        start = end - overlap
    return chunks


if __name__ == "__main__":
    import sys
    from loader import load_pdf

    pages = load_pdf(sys.argv[1])
    chunks = chunk_pages(pages)
    print("Total chunks:", len(chunks))
    for c in chunks[:3]:
        print("--- Page", c["page"], "| words:", len(c["text"].split()))
        print(c["text"][:200])