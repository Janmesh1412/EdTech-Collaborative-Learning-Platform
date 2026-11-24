# RAG Knowledge Assistant

A standalone Retrieval-Augmented Generation (RAG) mini-project inside the EdTech portfolio. Upload local UTF-8 Markdown/text files, search relevant chunks using pure-Python TF-IDF cosine similarity, and optionally generate a source-grounded answer with the OpenAI Responses API.

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

To enable generation, set the environment variable `OPENAI_API_KEY` before launching. Without it, the app remains a working document retrieval demo. API usage may incur charges.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Architecture

Upload (.txt/.md) → UTF-8 decoding → overlapping character chunks → TF-IDF ranking → top-k evidence → optional LLM response with numbered citations.

The retrieval implementation is intentionally dependency-light and auditable; it is lexical rather than embedding-based. Scores are relative similarity, not calibrated confidence. Model citations are requested but not independently verified; inspect evidence before relying on answers.

## Privacy and limitations

- Max 1 MB per file; uploaded text remains in app process memory.
- With generation enabled, selected source passages are transmitted to OpenAI.
- Treat uploaded text as untrusted. The prompt tells the model not to obey instructions in source passages; this is not a complete prompt-injection defense.
- No authentication, persistence, vector database, or PDF extraction in this starter.
- Do not upload confidential or sensitive documents to an untrusted deployment.
