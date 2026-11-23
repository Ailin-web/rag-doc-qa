import sys

import streamlit as st

sys.path.append("rag")

from generate import answer
from store import load_index

REFUSAL = "I couldn't find that in the documents."

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A")
st.caption("Ask a question about your notes. Answers come only from the documents, with page citations.")


@st.cache_resource
def get_index():
    return load_index()


try:
    chunks, vectors = get_index()
except FileNotFoundError:
    st.error("No index found. Run `python rag\\store.py` first.")
    st.stop()

question = st.text_input("Your question", placeholder="e.g. What is a kernel?")

if question.strip():
    with st.spinner("Searching the documents..."):
        try:
            text, sources = answer(question, chunks, vectors)
        except Exception as e:
            st.error("Something went wrong: " + str(e)[:150])
            st.stop()

    st.subheader("Answer")
    st.write(text)

    if REFUSAL not in text:
        with st.expander("Sources"):
            for s in sources:
                st.write("-", s)
elif question:
    st.warning("Please type a question.")