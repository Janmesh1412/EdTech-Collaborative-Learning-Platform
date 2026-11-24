"""Streamlit UI for a local-first RAG knowledge assistant."""
import os
import streamlit as st
from rag import prepare_documents, retrieve, grounded_answer

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚")
st.title("📚 RAG Knowledge Assistant")
st.caption("Upload .txt or .md files, retrieve supporting passages, and optionally generate a cited answer.")
uploads = st.file_uploader("Knowledge base files", type=["txt", "md"], accept_multiple_files=True)
question = st.text_input("Ask a question about your files")
top_k = st.slider("Passages to retrieve", 1, 8, 3)
documents = {}
for file in uploads or []:
    raw = file.getvalue()
    if len(raw) > 1_000_000:
        st.warning(f"Skipped {file.name}: file exceeds 1 MB.")
        continue
    try:
        documents[file.name] = raw.decode("utf-8")
    except UnicodeDecodeError:
        st.warning(f"Skipped {file.name}: not UTF-8.")
if st.button("Search knowledge base", type="primary"):
    if not question.strip():
        st.warning("Enter a question.")
    elif not documents:
        st.warning("Upload at least one valid document.")
    else:
        matches = retrieve(question, prepare_documents(documents), limit=top_k)
        st.subheader("Retrieved evidence")
        if not matches:
            st.info("No relevant passages found.")
        for i, match in enumerate(matches, 1):
            with st.expander(f"[{i}] {match['source']} · chunk {match['chunk']} · score {match['score']}", expanded=i == 1):
                st.write(match["text"])
        if matches and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                st.subheader("Grounded answer")
                st.write(grounded_answer(question, matches, OpenAI()))
            except Exception as exc:
                st.error(f"Generation failed: {type(exc).__name__}. Check your API configuration.")
        elif matches:
            st.info("Set OPENAI_API_KEY to enable AI-generated answers. Retrieval works without an API key.")
st.caption("Uploaded text is held in app memory. If generation is enabled, retrieved passages are sent to the configured OpenAI API.")
