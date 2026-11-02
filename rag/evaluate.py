"""Scoring retrieval and answers against the reviewed question set (eval/questions.csv)."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.csv"


def load_questions(path=QUESTIONS):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["answerable"] = r["answerable"].strip().lower() == "yes"
        r["pages"] = {int(p) for p in r["gold_pages"].split(";") if p.strip()}
        r["facts"] = [[normalise(alt.strip()) for alt in f.split("|")] for f in r["key_facts"].split(";") if f.strip()]
    return rows


def is_gold(chunk, q):
    """A passage counts as correct if it comes from the gold document and overlaps a gold page."""
    return (chunk["doc_id"] == q["gold_doc"]
            and any(p in q["pages"] for p in range(chunk["page_start"], chunk["page_end"] + 1)))


def retrieval_scores(questions, chunks, retriever, ks=(1, 3, 5)):
    """Hit@k (a gold passage in the top k), doc hit@5 and mean reciprocal rank, answerable questions only."""
    rows = []
    for q in questions:
        if not q["answerable"]:
            continue
        ranked = [chunks[i] for i, _ in retriever.search(q["question"], k=10)]
        rank = next((n for n, c in enumerate(ranked, 1) if is_gold(c, q)), None)
        rows.append({"id": q["id"], "rank": rank,
                     "doc_hit5": any(c["doc_id"] == q["gold_doc"] for c in ranked[:5])})
    n = len(rows)
    out = {f"hit@{k}": sum(r["rank"] is not None and r["rank"] <= k for r in rows) / n for k in ks}
    out["doc_hit@5"] = sum(r["doc_hit5"] for r in rows) / n
    out["mrr@10"] = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    out["n"] = n
    return out, rows


DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
                        "\u00a0": " ", "\u202f": " ", "\u2019": "'", "\u2018": "'"})


def normalise(text):
    """Lower-case and replace typographic dashes, quotes and non-breaking spaces with plain ones."""
    return text.translate(DASHES).lower()


def fact_coverage(answer, facts):
    text = normalise(answer)
    found = [any(alt in text for alt in alts) for alts in facts]
    return sum(found) / len(found) if found else 0.0


def answer_scores(questions, results):
    """results: {question id: answer dict from rag.answer.answer}."""
    rows = []
    for q in questions:
        r = results[q["id"]]
        if q["answerable"]:
            cov = fact_coverage(r["answer"], q["facts"]) if r["covered"] else 0.0
            rows.append({"id": q["id"], "answerable": True, "refused": not r["covered"],
                         "fact_coverage": cov, "all_facts": cov == 1.0,
                         "cited_gold": any(is_gold(s, q) for s in r["sources"]),
                         "cited_only_gold_doc": bool(r["sources"]) and all(s["doc_id"] == q["gold_doc"] for s in r["sources"])})
        else:
            rows.append({"id": q["id"], "answerable": False, "refused": not r["covered"]})
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    summary = {
        "answerable_n": len(ans),
        "answered": sum(not r["refused"] for r in ans) / len(ans),
        "all_key_facts": sum(r["all_facts"] for r in ans) / len(ans),
        "mean_fact_coverage": sum(r["fact_coverage"] for r in ans) / len(ans),
        "cited_gold_passage": sum(r["cited_gold"] for r in ans) / len(ans),
        "unanswerable_n": len(un),
        "correctly_refused": (sum(r["refused"] for r in un) / len(un)) if un else float("nan"),
    }
    return summary, rows
