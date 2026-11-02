"""Evaluate the assistant on the reviewed question set.

    python3 scripts/evaluate.py --retrieval-only   # search methods only (BM25 needs no model)
    python3 scripts/evaluate.py                    # + answers from the local model (needs Ollama)

Writes results/retrieval.csv, results/answers.csv (with a column for human review) and results/summary.md.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import evaluate as E, ollama  # noqa: E402
from rag.answer import answer  # noqa: E402
from rag.ingest import load_chunks  # noqa: E402
from rag.retrieve import BM25, EmbeddingIndex, Hybrid, load_embeddings  # noqa: E402

RES = ROOT / "results"
TOP_K = 5


def retrievers(chunks):
    out = {"BM25 (keywords)": BM25(chunks)}
    try:
        vecs, meta = load_embeddings(chunks)
    except FileNotFoundError:
        print("No embeddings yet (run scripts/embed.py): evaluating BM25 only.")
        return out
    emb = EmbeddingIndex(vecs, lambda q: ollama.embed([q], model=meta["model"])[0])
    out[f"Embeddings ({meta['model']})"] = emb
    out["Hybrid (BM25 + embeddings)"] = Hybrid(out["BM25 (keywords)"], emb)
    return out


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N questions (quick trial)")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    chunks = load_chunks()
    questions = E.load_questions()[:a.limit] if a.limit else E.load_questions()

    lines = ["# Evaluation summary", "", f"{len(chunks)} passages from 12 MDCG documents; "
             f"{sum(q['answerable'] for q in questions)} answerable and "
             f"{sum(not q['answerable'] for q in questions)} unanswerable questions.", "",
             "## 1. Retrieval: is a correct passage (right document and page) found?", "",
             "| Method | Hit@1 | Hit@3 | Hit@5 | Right document in top 5 | MRR@10 |", "|---|---|---|---|---|---|"]
    rets = retrievers(chunks)
    per_q = {}
    scores = {}
    for name, r in rets.items():
        s, rows = E.retrieval_scores(questions, chunks, r)
        scores[name] = s
        lines.append(f"| {name} | {pct(s['hit@1'])} | {pct(s['hit@3'])} | {pct(s['hit@5'])} | "
                     f"{pct(s['doc_hit@5'])} | {s['mrr@10']:.2f} |")
        for row in rows:
            per_q.setdefault(row["id"], {})[name] = row["rank"]
        print(f"{name}: hit@5 {pct(s['hit@5'])}")
    with open(RES / "retrieval.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id"] + [f"rank: {n}" for n in rets])
        for qid, ranks in per_q.items():
            w.writerow([qid] + [ranks.get(n) or ">10" for n in rets])
    (RES / "retrieval.json").write_text(json.dumps(scores, indent=2))

    if not a.retrieval_only:
        best = max(scores, key=lambda n: (scores[n]["hit@5"], scores[n]["mrr@10"]))
        retriever = rets[best]
        print(f"\nAnswering {len(questions)} questions with {ollama.CHAT_MODEL}, passages from {best} ...")
        results, t0 = {}, time.perf_counter()
        for n, q in enumerate(questions, 1):
            top = [chunks[i] for i, _ in retriever.search(q["question"], k=TOP_K)]
            try:
                results[q["id"]] = answer(q["question"], top)
            except ollama.OllamaError as exc:
                results[q["id"]] = {"question": q["question"], "covered": False, "answer": f"ERROR: {exc}",
                                    "sources": [], "usage": {"seconds": 0}}
            r = results[q["id"]]
            print(f"  [{n}/{len(questions)}] {q['id']} {'answered' if r['covered'] else 'not covered'} "
                  f"({r['usage'].get('seconds', 0)} s)", flush=True)
        summ, rows = E.answer_scores(questions, results)
        secs = [r["usage"].get("seconds", 0) for r in results.values()]
        with open(RES / "answers.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "question", "answerable", "model_answer", "sources", "key_fact_coverage",
                        "cited_gold_passage", "gold_answer", "human_verdict (correct / partly / wrong)",
                        "human_comment"])
            gold = {q["id"]: q for q in questions}
            for row in rows:
                r, q = results[row["id"]], gold[row["id"]]
                w.writerow([row["id"], q["question"], "yes" if q["answerable"] else "no", r["answer"],
                            "; ".join(f"{s['doc_id']} p{s['page_start']}" for s in r["sources"]),
                            f"{row.get('fact_coverage', ''):.2f}" if q["answerable"] else "",
                            row.get("cited_gold", ""), q["gold_answer"], "", ""])
        (RES / "answers.json").write_text(json.dumps(
            {"model": ollama.CHAT_MODEL, "retriever": best, "top_k": TOP_K, "summary": summ,
             "results": {k: {**v, "sources": [s["chunk_id"] for s in v["sources"]]} for k, v in results.items()}},
            indent=2))
        lines += ["", f"## 2. Answers from {ollama.CHAT_MODEL} (top {TOP_K} passages from {best})", "",
                  "| Measure | Result |", "|---|---|",
                  f"| Answerable questions answered (not refused) | {pct(summ['answered'])} |",
                  f"| Answers containing all key facts | {pct(summ['all_key_facts'])} |",
                  f"| Average share of key facts present | {pct(summ['mean_fact_coverage'])} |",
                  f"| Answers citing a correct passage | {pct(summ['cited_gold_passage'])} |",
                  f"| Unanswerable questions correctly refused | {pct(summ['correctly_refused'])} "
                  f"({summ['unanswerable_n']} questions) |",
                  f"| Average time per answer | {sum(secs) / len(secs):.1f} s |", "",
                  "Key facts are checked automatically by matching words; a human review of every answer is "
                  "recorded in `results/answers.csv`."]
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
