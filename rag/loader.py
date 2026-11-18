import pymupdf


def load_pdf(path):
    """Return a list of {'page': number, 'text': text} for each page."""
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({"page": i, "text": text})
    if not pages:
        raise ValueError("No text found. This may be a scanned PDF.")
    return pages


if __name__ == "__main__":
    import sys

    result = load_pdf(sys.argv[1])
    print("Pages with text:", len(result))
    print("--- First page preview ---")
    print(result[0]["text"][:500])