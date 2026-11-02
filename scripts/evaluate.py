"""Evaluate the assistant on the reviewed question set.

    python3 scripts/evaluate.py --retrieval-only               # search methods only (BM25 needs no model)
    python3 scripts/evaluate.py --tag v1-top8 --prompt v1 --top-k 8  # + answers from the local model (needs Ollama)
    python3 scripts/evaluate.py --rescore v1                    # re-score saved answers, no model needed

Retrieval results go to results/retrieval.*; each answer run goes to results/<tag>/ (answers.csv with a
column for human review, answers.json, summary.md), so earlier runs are kept for comparison.
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
from rag.answer import DEFAULT_PROMPT, DEFAULT_TOP_K, PROMPTS, answer  # noqa: E402
from rag.ingest import load_chunks  # noqa: E402
from rag.retrieve import BM25, EmbeddingIndex, Hybrid, load_embeddings  # noqa: E402

RES = ROOT / "results"


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


def evaluate_retrieval(questions, chunks, rets):
    lines = ["## Retrieval: is a correct passage (right document and page) found?", "",
             "| Method | Hit@1 | Hit@3 | Hit@5 | Hit@8 | Right document in top 5 | MRR@10 |",
             "|---|---|---|---|---|---|---|"]
    per_q, scores = {}, {}
    for name, r in rets.items():
        s, rows = E.retrieval_scores(questions, chunks, r, ks=(1, 3, 5, 8))
        scores[name] = s
        lines.append(f"| {name} | {pct(s['hit@1'])} | {pct(s['hit@3'])} | {pct(s['hit@5'])} | {pct(s['hit@8'])} | "
                     f"{pct(s['doc_hit@5'])} | {s['mrr@10']:.2f} |")
        for row in rows:
            per_q.setdefault(row["id"], {})[name] = row["rank"]
    with open(RES / "retrieval.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id"] + [f"rank: {n}" for n in rets])
        for qid, ranks in per_q.items():
            w.writerow([qid] + [ranks.get(n) or ">10" for n in rets])
    (RES / "retrieval.json").write_text(json.dumps(scores, indent=2))
    (RES / "retrieval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return scores


def write_answer_outputs(out_dir, questions, results, config):
    out_dir.mkdir(parents=True, exist_ok=True)
    summ, rows = E.answer_scores(questions, results)
    secs = [r["usage"].get("seconds", 0) for r in results.values()]
    gold = {q["id"]: q for q in questions}
    with open(out_dir / "answers.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "question", "answerable", "model_answer", "sources", "key_fact_coverage",
                    "cited_gold_passage", "gold_answer", "human_verdict (correct / partly / wrong)", "human_comment"])
        for row in rows:
            r, q = results[row["id"]], gold[row["id"]]
            w.writerow([row["id"], q["question"], "yes" if q["answerable"] else "no", r["answer"],
                        "; ".join(f"{s['doc_id']} p{s['page_start']}" for s in r["sources"]),
                        f"{row['fact_coverage']:.2f}" if q["answerable"] else "",
                        row.get("cited_gold", ""), q["gold_answer"], "", ""])
    (out_dir / "answers.json").write_text(json.dumps(
        {**config, "summary": summ,
         "results": {k: {**v, "sources": [s["chunk_id"] for s in v["sources"]]} for k, v in results.items()}},
        indent=2))
    lines = [f"## Answers: run {config['tag']}", "",
             f"Model {config['model']}, prompt {config['prompt']}, top {config['top_k']} passages from "
             f"{config['retriever']}.", "", "| Measure | Result |", "|---|---|",
             f"| Answerable questions answered (not refused) | {pct(summ['answered'])} |",
             f"| Answers containing all key facts (automatic check) | {pct(summ['all_key_facts'])} |",
             f"| Average share of key facts present | {pct(summ['mean_fact_coverage'])} |",
             f"| Answers citing a correct passage | {pct(summ['cited_gold_passage'])} |",
             f"| Unanswerable questions correctly refused | {pct(summ['correctly_refused'])} "
             f"({summ['unanswerable_n']} questions) |",
             f"| Average time per answer | {sum(secs) / len(secs):.1f} s |"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return summ


def rescore(tag, questions, chunks):
    saved = json.loads((RES / tag / "answers.json").read_text())
    by_id = {c["chunk_id"]: c for c in chunks}
    results = {k: {**v, "sources": [by_id[c] for c in v["sources"]]} for k, v in saved["results"].items()}
    config = {k: saved[k] for k in ("tag", "model", "prompt", "top_k", "retriever")}
    return write_answer_outputs(RES / tag, questions, results, config)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--rescore", metavar="TAG", help="re-score a saved run without calling the model")
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--prompt", choices=list(PROMPTS), default=DEFAULT_PROMPT)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--retriever", default="Hybrid (BM25 + embeddings)")
    ap.add_argument("--limit", type=int, help="only the first N questions (quick trial)")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    chunks = load_chunks()
    questions = E.load_questions()[:a.limit] if a.limit else E.load_questions()
    if a.rescore:
        return rescore(a.rescore, questions, chunks)

    rets = retrievers(chunks)
    evaluate_retrieval(questions, chunks, rets)
    if a.retrieval_only:
        return
    name = a.retriever if a.retriever in rets else "BM25 (keywords)"
    retriever = rets[name]
    print(f"\nRun {a.tag}: answering {len(questions)} questions with {ollama.CHAT_MODEL}, prompt {a.prompt}, "
          f"top {a.top_k} passages from {name} ...")
    results = {}
    for n, q in enumerate(questions, 1):
        top = [chunks[i] for i, _ in retriever.search(q["question"], k=a.top_k)]
        try:
            results[q["id"]] = answer(q["question"], top, prompt=a.prompt)
        except ollama.OllamaError as exc:
            results[q["id"]] = {"question": q["question"], "covered": False, "answer": f"ERROR: {exc}",
                                "sources": [], "usage": {"seconds": 0}}
        r = results[q["id"]]
        print(f"  [{n}/{len(questions)}] {q['id']} {'answered' if r['covered'] else 'not covered'} "
              f"({r['usage'].get('seconds', 0)} s)", flush=True)
    write_answer_outputs(RES / a.tag, questions, results,
                         {"tag": a.tag, "model": ollama.CHAT_MODEL, "prompt": a.prompt, "top_k": a.top_k,
                          "retriever": name})


if __name__ == "__main__":
    main()
