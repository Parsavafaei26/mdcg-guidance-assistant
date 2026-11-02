"""MDCG guidance assistant.   Run locally:  streamlit run streamlit_app.py"""
import csv
import html
import json
import re
from pathlib import Path

import streamlit as st

from rag import ollama
from rag.answer import answer
from rag.evaluate import load_questions
from rag.ingest import load_chunks, load_manifest
from rag.retrieve import BM25, EmbeddingIndex, Hybrid, load_embeddings, tokens

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="MDCG Guidance Assistant", page_icon="📘", layout="wide")


@st.cache_resource
def setup():
    chunks = load_chunks()
    urls = {m["doc_id"]: m["url"] for m in load_manifest()}
    return chunks, BM25(chunks), urls


@st.cache_resource
def local_retriever(_chunks, _bm25):
    """Hybrid search when the embeddings and a local Ollama are available, otherwise None."""
    if ollama.available() is None:
        return None
    try:
        vecs, meta = load_embeddings(_chunks)
    except (FileNotFoundError, ValueError):
        return None
    return Hybrid(_bm25, EmbeddingIndex(vecs, lambda q: ollama.embed([q], model=meta["model"])[0]))


chunks, bm25, urls = setup()


def highlight(text, query):
    words = {w for w in tokens(query) if len(w) > 2}
    safe = html.escape(text)
    for w in sorted(words, key=len, reverse=True):
        safe = re.sub(rf"(?i)\b({re.escape(w)}\w*)", r"<mark>\1</mark>", safe)
    return safe


def show_passage(n, c, query=""):
    pages = f"p. {c['page_start']}" + ("" if c["page_end"] == c["page_start"] else f"-{c['page_end']}")
    link = f"{urls[c['doc_id']]}#page={c['page_start']}"
    st.markdown(f"**[{n}] {c['doc_id']}: {c['title']}**, {pages} · [open the PDF at this page]({link})")
    st.markdown(f"<div style='font-size:0.92em;line-height:1.5'>{highlight(c['text'], query)}</div>",
                unsafe_allow_html=True)
    st.divider()


st.title("MDCG guidance assistant")
st.caption("Search and question-answering over 12 EU MDCG guidance documents (MDR/IVDR) · every answer cites "
           "document and page · educational project; always check the official text")

tab1, tab2, tab3 = st.tabs(["Search the guidance", "Ask a question", "How well does it work?"])

with tab1:
    q = st.text_input("Search", placeholder="e.g. PSUR frequency for class IIb devices", key="search")
    if q:
        for n, (i, _) in enumerate(bm25.search(q, k=5), 1):
            show_passage(n, chunks[i], q)
    st.caption("Keyword search (BM25) over the passages. It runs anywhere, including this online demo.")

with tab2:
    retriever = local_retriever(chunks, bm25)
    if retriever is None:
        st.info("The answering model runs privately on the author's laptop (an open-source model through "
                "Ollama), so it is not available in this online demo. Below are the answers it gave to the "
                "test questions, exactly as produced.")
        res_file = ROOT / "results" / "answers.json"
        if res_file.exists():
            saved = json.loads(res_file.read_text())
            qs = {q["id"]: q for q in load_questions()}
            by_id = {c["chunk_id"]: c for c in chunks}
            pick = st.selectbox("Test question", list(saved["results"]),
                                format_func=lambda k: f"{k}: {qs[k]['question']}")
            r = saved["results"][pick]
            st.markdown(f"**Answer** ({saved['model']}):  \n{r['answer']}")
            for n, cid in enumerate(r["sources"], 1):
                show_passage(n, by_id[cid])
            st.markdown(f"**Reviewed reference answer:** {qs[pick]['gold_answer']}")
        else:
            st.warning("No saved answers yet: run scripts/evaluate.py on a computer with Ollama.")
    else:
        question = st.text_input("Your question", placeholder="e.g. Who uploads the SSCP to Eudamed?")
        if question:
            with st.spinner("Searching and answering locally ..."):
                top = [chunks[i] for i, _ in retriever.search(question, k=5)]
                r = answer(question, top)
            st.markdown(f"**Answer:**  \n{r['answer']}")
            for n, c in enumerate(r["sources"], 1):
                show_passage(n, c, question)
            st.caption(f"{ollama.CHAT_MODEL} on this computer · {r['usage'].get('seconds', 0)} s")

with tab3:
    summary = ROOT / "results" / "summary.md"
    st.markdown(summary.read_text().replace("# Evaluation summary", "") if summary.exists()
                else "No results yet.")
    st.markdown("Method, limits and the reviewed question set are described in the "
                "[README](https://github.com/Parsavafaei26/mdcg-guidance-assistant).")

with st.sidebar:
    st.subheader("Documents searched")
    for m in load_manifest():
        st.markdown(f"[{m['doc_id']}]({m['url']}): {m['title']} ({m['version']})")
    st.caption("MDCG guidance is not legally binding; only the regulations and the Court of Justice are.")
