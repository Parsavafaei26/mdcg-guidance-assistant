"""Tests:  python -m pytest -q   (no model or internet needed)"""
import io
import json
from pathlib import Path

import numpy as np
import pytest

from rag import evaluate as E, ollama
from rag.answer import NOT_COVERED, answer
from rag.ingest import clean_pages, is_heading_line, is_next_heading, load_chunks, load_manifest, paragraphs
from rag.retrieve import BM25, EmbeddingIndex, Hybrid, rrf, tokens

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


# ---------------------------------------------------------------- ingestion
def test_every_document_is_indexed(chunks):
    assert {c["doc_id"] for c in chunks} == {m["doc_id"] for m in load_manifest()}


def test_chunks_are_well_formed(chunks):
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert 1 <= c["page_start"] <= c["page_end"]
        assert len(c["text"].split()) >= 25
        assert c["text"].count("....") < 3, "table-of-contents text should be dropped"


def test_running_headers_and_page_numbers_removed():
    pages = [f"Medical Devices\nMDCG 2099-1\n{n}\nBody text on page {n}.\nPage {n} of 5" for n in range(1, 6)]
    cleaned = clean_pages(pages)
    assert cleaned[0] == ["Body text on page 1."]


def test_heading_detection_rejects_footnotes_and_contents():
    assert is_heading_line("3 Equivalence")
    assert not is_heading_line("13 In this context the terms are the same.")
    assert not is_heading_line("2 Scope ........ 4")
    assert is_next_heading((2,), (3,)) and is_next_heading((2,), (2, 1)) and is_next_heading((2, 3), (3,))
    assert not is_next_heading((2,), (8,))


def test_paragraphs_join_wrapped_lines():
    lines = ["This guidance applies to all", "medical devices.", "• first point", "2 Scope", "Text"]
    assert paragraphs(lines) == ["This guidance applies to all medical devices.", "• first point", "2 Scope", "Text"]


# ---------------------------------------------------------------- retrieval
def test_tokens():
    assert tokens("The devices of Class IIb (MDR 2017/745)") == ["device", "class", "iib", "mdr", "2017/745"]


def test_bm25_ranks_the_matching_text_first():
    docs = [{"doc_id": "A", "title": "t", "text": "periodic safety update report frequency"},
            {"doc_id": "B", "title": "t", "text": "classification rules for software"},
            {"doc_id": "C", "title": "t", "text": "cybersecurity of networks"}]
    assert BM25(docs).search("software classification", k=1)[0][0] == 1


def test_rrf_rewards_agreement():
    fused = rrf([[(1, 9), (2, 8)], [(2, 9), (3, 8)]], k=3)
    assert fused[0][0] == 2


def test_embedding_and_hybrid_search():
    vecs = np.eye(3)
    emb = EmbeddingIndex(vecs, lambda q: [0, 1, 0])
    assert emb.search("anything", k=1)[0][0] == 1
    docs = [{"doc_id": d, "title": "t", "text": t} for d, t in [("A", "alpha"), ("B", "beta"), ("C", "gamma")]]
    assert Hybrid(BM25(docs), emb).search("beta", k=1)[0][0] == 1


def test_bm25_finds_the_reviewed_answers(chunks):
    """Regression guard: keyword search keeps finding a gold passage in the top 5 for most questions."""
    s, _ = E.retrieval_scores(E.load_questions(), chunks, BM25(chunks))
    assert s["hit@5"] >= 0.85 and s["doc_hit@5"] >= 0.95


# ---------------------------------------------------------------- question set
def test_question_set_is_consistent(chunks):
    qs = E.load_questions()
    assert len({q["id"] for q in qs}) == len(qs)
    docs = {m["doc_id"] for m in load_manifest()}
    for q in qs:
        if q["answerable"]:
            assert q["gold_doc"] in docs and q["pages"] and q["facts"]
            # every key fact really is on the gold pages
            text = " ".join(c["text"] for c in chunks if E.is_gold(c, q)).lower()
            for alts in q["facts"]:
                assert any(a in text for a in alts), (q["id"], alts)


def test_fact_coverage():
    assert E.fact_coverage("Report within 15 days, or 2 days", [["15 days"], ["2 days"], ["10 days"]]) == 2 / 3


# ---------------------------------------------------------------- answering
CH = [{"chunk_id": "X#1", "doc_id": "MDCG 2023-3", "title": "Vigilance", "page_start": 8, "page_end": 8,
       "text": "no later than 15 days"},
      {"chunk_id": "X#2", "doc_id": "MDCG 2022-21", "title": "PSUR", "page_start": 9, "page_end": 9, "text": "x"}]


def test_answer_maps_citations_to_sources():
    fake = lambda system, user, schema, **kw: ({"covered": True, "answer": "15 days [1]", "citations": [1, 7]},
                                              {"seconds": 1})
    r = answer("deadline?", CH, chat=fake)
    assert r["covered"] and [s["chunk_id"] for s in r["sources"]] == ["X#1"] and r["invalid_citations"] == [7]


def test_answer_without_valid_citation_counts_as_not_covered():
    fake = lambda *a, **kw: ({"covered": True, "answer": "made up", "citations": []}, {"seconds": 1})
    r = answer("q", CH, chat=fake)
    assert not r["covered"] and r["answer"] == NOT_COVERED


def test_prompt_contains_numbered_passages_and_rules():
    seen = {}

    def fake(system, user, schema, **kw):
        seen.update(system=system, user=user, schema=schema)
        return {"covered": False, "answer": "", "citations": []}, {}
    answer("q?", CH, chat=fake)
    assert "[1] MDCG 2023-3 (Vigilance), page 8" in seen["user"] and "ONLY" in seen["system"]
    assert seen["schema"]["required"] == ["covered", "answer", "citations"]


def test_answer_scores():
    qs = [{"id": "A", "answerable": True, "gold_doc": "MDCG 2023-3", "pages": {8}, "facts": [["15 days"]]},
          {"id": "U", "answerable": False}]
    res = {"A": {"covered": True, "answer": "15 days", "sources": [CH[0]]},
           "U": {"covered": False, "answer": NOT_COVERED, "sources": []}}
    s, _ = E.answer_scores(qs, res)
    assert s["all_key_facts"] == 1 and s["cited_gold_passage"] == 1 and s["correctly_refused"] == 1


# ---------------------------------------------------------------- ollama client (simulated)
class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ollama_chat_sends_schema_and_parses_reply():
    sent = {}

    def opener(req, timeout=None):
        sent.update(json.loads(req.data))
        return Resp(json.dumps({"message": {"content": '{"covered": false, "answer": "", "citations": []}'},
                                "prompt_eval_count": 10, "eval_count": 5, "done_reason": "stop"}).encode())
    out, usage = ollama.chat_json("sys", "user", {"type": "object"}, opener=opener)
    assert out["covered"] is False and sent["options"]["temperature"] == 0 and sent["format"] == {"type": "object"}


def test_ollama_embed_adds_task_prefix():
    sent = {}

    def opener(req, timeout=None):
        sent.update(json.loads(req.data))
        return Resp(json.dumps({"embeddings": [[0.1, 0.2]]}).encode())
    ollama.embed(["PSUR"], model="nomic-embed-text", kind="query", opener=opener)
    assert sent["input"] == ["search_query: PSUR"]


# ---------------------------------------------------------------- app
def test_app_runs_without_a_local_model():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    at.text_input(key="search").input("PSUR frequency class IIb").run()
    assert not at.exception
    assert any("MDCG 2022-21" in m.value for m in at.markdown)
