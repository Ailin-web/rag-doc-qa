import os
import sys

import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, "rag"))

from generate import answer
from store import load_index
from upload import index_uploaded_pdf

REFUSAL = "I couldn't find that in the documents."

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A")
st.caption("Ask a question about your documents. Answers come only from the documents, with page citations.")


@st.cache_resource(show_spinner="Loading the built-in document...")
def get_index():
    # 1) a prebuilt index (your own computer)
    try:
        return load_index()
    except FileNotFoundError:
        pass
    # 2) a small demo PDF shipped with the app (the cloud)
    demo = os.path.join(BASE, "demo", "sample.pdf")
    if os.path.exists(demo):
        with open(demo, "rb") as f:
            return index_uploaded_pdf(f.read(), "sample.pdf")
    # 3) nothing built in: visitors upload their own PDF
    return None, None


try:
    base_chunks, base_vectors = get_index()
except Exception as e:
    base_chunks, base_vectors = None, None
    st.warning("Could not load the built-in document (" + str(e)[:100] + "). You can still upload a PDF.")

ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("uploaded", None)

with st.sidebar:
    st.header("Documents")
    up = st.file_uploader("Upload your own PDF", type="pdf")
    if up is not None:
        key = (up.name, up.size)
        if ss.uploaded is None or ss.uploaded["key"] != key:
            with st.spinner("Reading and indexing your PDF..."):
                try:
                    c, v = index_uploaded_pdf(up.getvalue(), up.name)
                    ss.uploaded = {"key": key, "name": up.name, "chunks": c, "vectors": v}
                    ss.history = []
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error("Could not read this PDF: " + str(e)[:100])
    elif ss.uploaded is not None:
        ss.uploaded = None
        ss.history = []

    if ss.uploaded:
        chunks, vectors = ss.uploaded["chunks"], ss.uploaded["vectors"]
        st.success("Using your PDF: " + ss.uploaded["name"])
    elif base_chunks is not None:
        chunks, vectors = base_chunks, base_vectors
        st.write("Using the built-in document(s)")
    else:
        chunks, vectors = None, None

    if chunks is not None:
        st.write("Documents indexed:", len({c["source"] for c in chunks}))
        st.write("Chunks:", len(chunks))
    if st.button("Clear chat"):
        ss.history = []
        st.rerun()

if chunks is None:
    st.info("👈 Upload a PDF in the sidebar to start asking questions.")
    st.stop()


def show_answer(text, sources):
    st.write(text)
    if sources and REFUSAL not in text:
        with st.expander("Sources"):
            for s in sources:
                st.write("-", s)


for item in ss.history:
    with st.chat_message("user"):
        st.write(item["q"])
    with st.chat_message("assistant"):
        show_answer(item["a"], item["sources"])

question = st.chat_input("Ask a question about your documents")

if question and question.strip():
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            try:
                text, sources = answer(question, chunks, vectors)
            except Exception as e:
                text = "Something went wrong, please try again. (" + str(e)[:100] + ")"
                sources = []
        show_answer(text, sources)
    ss.history.append({"q": question, "a": text, "sources": sources})