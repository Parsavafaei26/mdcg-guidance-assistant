# MDCG guidance assistant: search and cited answers over EU medical device guidance

[![tests](https://github.com/Parsavafaei26/mdcg-guidance-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/Parsavafaei26/mdcg-guidance-assistant/actions/workflows/tests.yml)

A retrieval-augmented generation (RAG) assistant over 12 EU MDCG guidance documents (MDR/IVDR). It finds
the relevant passages, answers only from them, cites document and page, and says when the guidance does not
cover a question. The language model runs free and offline on a laptop, and every answer was checked by hand
against a set of 46 reviewed regulatory questions.

**Main result:** with the adopted setup, 32 of 40 answerable questions were answered correctly, 5 partly,
none wrongly, and 3 were refused; all 6 questions outside the guidance were correctly refused. Reading the
answers mattered: the automatic score could not tell the setups apart, and an instruction change that looked
sensible added two wrong answers.

## What it does

```
question ──► search 505 passages ──► top 8 passages ──► local LLM ──► answer + citations
             (keywords + meaning)                       (JSON: covered, answer, citations)
                                                                 │
                                       no valid citation ────────┴──► "The provided MDCG guidance does not cover this."
```

1. **Ingestion** (`rag/ingest.py`). Each PDF is read page by page (`pdftotext`). Running headers, page
   numbers and contents pages are removed, and the text is split into passages of about 220 words with one
   paragraph of overlap. Every passage keeps its document number and page range: 505 passages in total.
2. **Search** (`rag/retrieve.py`). Three methods, all written for this project and compared on the same questions:
   - **BM25** keyword search, with a small tokeniser for regulatory text (keeps "IIb", "2017/745").
   - **Embeddings** (`nomic-embed-text`) with cosine similarity, which finds passages with the same meaning in different words.
   - **Hybrid**: both rankings combined by reciprocal rank fusion.
3. **Answering** (`rag/answer.py`). The model receives the numbered passages and strict rules: use only the
   passages, cite them, keep numbers and device classes exactly as written, and refuse when the answer is not
   there. The reply is forced into a JSON schema (`covered`, `answer`, `citations`). An answer that cites no
   valid passage is treated as a refusal, so the app never shows an unsupported answer.
4. **App** (`streamlit_app.py`). Search the guidance with highlighted matches and a link that opens the
   official PDF at the right page; ask a question (on a computer running Ollama); see the evaluation.

## Documents

| Document | Topic |
|---|---|
| MDCG 2019-9 | Summary of safety and clinical performance (SSCP) |
| MDCG 2019-11 | Qualification and classification of software (MDSW) |
| MDCG 2019-16 | Cybersecurity for medical devices |
| MDCG 2020-5 | Clinical evaluation: equivalence |
| MDCG 2020-6 | Sufficient clinical evidence for legacy devices |
| MDCG 2020-7 | PMCF plan template |
| MDCG 2020-8 | PMCF evaluation report template |
| MDCG 2021-24 | Classification of medical devices (Rev.1) |
| MDCG 2021-25 | MDR requirements for legacy devices |
| MDCG 2022-21 | Periodic safety update report (PSUR) |
| MDCG 2023-3 | Vigilance terms and concepts (Rev.2) |
| MDCG 2025-10 | Post-market surveillance |

Exact versions and download links are in `data/manifest.csv`. The PDFs are not stored in this repository:
download them from those links into `data/pdf/` to rebuild the passages.

## Test questions

`eval/questions.csv` holds 46 questions: 40 answerable from the documents and 6 that are not (notified body
fees, FDA 510(k), UKCA marking, Member State penalties, HIPAA, an ISO 13485 clause). Each answerable question has
a reference answer, the source document and page(s), and key facts. The questions were drafted with LLM
assistance and reviewed by the author against the PDFs; an automated test checks that every key fact really
appears on the stated pages.

## Results

### Search: is a correct passage (right document and page) found?

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@8 | Right document in top 5 | MRR@10 |
|---|---|---|---|---|---|---|
| BM25 (keywords) | 65% | 78% | 88% | 98% | 98% | 0.73 |
| Embeddings | 60% | 85% | 90% | 95% | 95% | 0.72 |
| **Hybrid** | **68%** | 85% | **92%** | **98%** | 95% | **0.78** |

Hybrid search was best overall. The two methods fail on different questions: keywords miss paraphrases
("duration-of-use categories" was not in the keyword top 10 but was first by meaning), while embeddings
miss exact terms such as document numbers.

### Answers: human review of every answer

Model `granite4.2:8b` (IBM Granite, open weights) through Ollama on a MacBook Pro (M3 Pro, 18 GB), temperature
0, about 16 seconds per answer, no cost. Three setups were run on all 46 questions:

| Verdict (40 answerable questions) | v1: prompt v1, top 5 | **v1-top8: prompt v1, top 8 (adopted)** | v2: prompt v2, top 8 |
|---|---|---|---|
| Correct | 29 | **32** | 32 |
| Partly correct | 7 | 5 | 4 |
| **Wrong** | 2 | **0** | 2 |
| Refused (answer was in the documents) | 2 | 3 | 2 |
| Unanswerable questions correctly refused | 6 of 6 | 6 of 6 | 6 of 6 |

Verdicts for all 138 answers are in `results/human_review.csv`, with a reason for every answer that is not
fully correct (drafted with LLM assistance, checked by the author). All model outputs are in `results/<run>/`.

**What happened, in order:**

1. **v1** (top 5 passages) gave two wrong answers. For the serious-incident reporting deadlines it said
   **2 days** for death or serious deterioration; the guidance says **10 days** (2 days is for a serious public
   health threat). On combining parts of different devices to claim equivalence it said "Yes"; the
   guidance says no, except in exceptional cases. Two other questions were refused because the right passage
   was ranked below 5th.
2. **v2** was written after reading v1: 8 passages instead of 5, plus two extra rules ("list every case with
   its own value, never merge them" and "start with Yes or No when that fits"). It fixed the equivalence
   answer and the 10-day deadline, but gave the public health threat deadline as 15 days and answered
   "Yes - class I devices do not need a PSUR".
3. **v1-top8** separates the two changes: the original instructions with 8 passages. It gave **no wrong
   answers**. The improvement came from giving the model more of the right text (hybrid hit@8 is 98% vs 92%
   at 5), and the extra rules made a small model less reliable rather than more. This setup was adopted.

**The automatic score was misleading.** A key-fact string match (`results/<run>/summary.md`) marked down
answers that were correct but worded differently ("highest" instead of "higher"), and did not penalise the
contradictory "Yes". It is kept as a quick regression check only.

Remaining weak points in the adopted setup: incomplete answers to broad questions (on who is responsible for
cybersecurity it misses "joint responsibility of all stakeholders"), one deadline case left out, three refusals
where the passage was available, and article numbers garbled by footnote marks in the PDF text
(Article 18 with footnote 8 becomes "Article 188").

## Limits

- **The setup was chosen on the same 46 questions** it is scored on; there is no separate held-out set, so the
  adopted result is likely optimistic. The questions were drafted while reading passages found by keyword
  search, which may favour BM25.
- 12 documents only; for example MDCG 2020-3 (significant changes) is not included. Guidance is revised, so
  the versions in `data/manifest.csv` matter.
- A small model running on a laptop. A larger model would probably refuse less, but the aim here is a free,
  private setup and an honest measurement of it.
- MDCG guidance is not legally binding; only the regulations and the Court of Justice are. This is an
  educational project, not regulatory advice: always check the official text.

## Run it

```bash
pip install -r requirements.txt
python3 -m rag.ingest                                # PDFs in data/pdf/ -> data/chunks.jsonl (needs poppler)
python3 scripts/evaluate.py --retrieval-only         # keyword search only, no model needed

# with Ollama (https://ollama.com): ollama pull nomic-embed-text && ollama pull granite4.2:8b
python3 scripts/embed.py                             # passage embeddings -> data/embeddings.npy
python3 scripts/evaluate.py --tag v1-top8            # answers -> results/v1-top8/
python3 scripts/evaluate.py --rescore v1-top8        # re-score saved answers without the model
python3 scripts/review_summary.py                    # results/human_review.csv -> human_review.md

streamlit run streamlit_app.py
pip install -r requirements-dev.txt && python3 -m pytest -q
```

The online demo has no language model, so it offers keyword search plus the saved answers and their review.

## Repository

```
rag/          ingest.py, retrieve.py, answer.py, ollama.py, evaluate.py
scripts/      embed.py, evaluate.py, review_summary.py
eval/         questions.csv (46 reviewed test questions)
data/         manifest.csv, chunks.jsonl, embeddings (PDFs not stored)
results/      retrieval, three answer runs, human_review.csv/.md
tests/        20 tests (no model or internet needed), run on every push by GitHub Actions
```

## Source

MDCG guidance documents © European Union, published by the European Commission
(health.ec.europa.eu). Reuse is authorised under Commission Decision 2011/833/EU provided the source is
acknowledged. Passages in `data/chunks.jsonl` are extracts of these documents.

---

**Author:** Parsa Vafaei, MSc Biomedical Engineering, University of Sheffield.
