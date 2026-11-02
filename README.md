# MDCG guidance assistant: search and cited answers over EU medical device guidance

[![tests](https://github.com/Parsavafaei26/mdcg-guidance-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/Parsavafaei26/mdcg-guidance-assistant/actions/workflows/tests.yml)

*Work in progress: evaluation results are added after the local model run and the review of the test questions.*

A retrieval-augmented generation (RAG) assistant over 12 EU MDCG guidance documents (MDR/IVDR):
it finds the relevant passages, answers only from them, cites document and page, and says when the
guidance does not cover a question. It runs on a free, local open-source model, and it is tested against
a set of reviewed regulatory questions.

## Run it

```bash
pip install -r requirements.txt
python3 scripts/evaluate.py --retrieval-only     # keyword search, no model needed
python3 scripts/embed.py                         # needs Ollama + nomic-embed-text
python3 scripts/evaluate.py                      # needs Ollama + a chat model (default granite4.2:8b)
streamlit run streamlit_app.py
pip install -r requirements-dev.txt && python3 -m pytest -q
```

---

**Author:** Parsa Vafaei, MSc Biomedical Engineering, University of Sheffield.
