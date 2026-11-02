"""Summarise the human review of the model's answers (results/human_review.csv -> results/human_review.md).

Every answer of every run was read and given a verdict: correct / partly / wrong / refused for the 40
answerable questions, correctly refused / wrongly answered for the 6 unanswerable ones.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RUNS = {"v1": "Prompt v1, top 5 passages",
        "v1-top8": "Prompt v1, top 8 passages (adopted)",
        "v2": "Prompt v2, top 8 passages"}
VERDICTS = ["correct", "partly", "wrong", "refused"]


def load(path=RES / "human_review.csv"):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def counts(rows):
    out = defaultdict(Counter)
    for r in rows:
        out[r["run"]][r["final_verdict"]] += 1
    return out


def main():
    rows = load()
    c = counts(rows)
    runs = [r for r in RUNS if r in c]
    n_ans = sum(c[runs[0]][v] for v in VERDICTS)
    n_un = sum(c[runs[0]][v] for v in ("correctly refused", "wrongly answered"))
    lines = ["## Answers: human review", "",
             f"Every answer was read and judged against the guidance ({n_ans} answerable and {n_un} unanswerable "
             "questions per run).", "",
             "| Verdict | " + " | ".join(f"{r}: {RUNS[r]}" for r in runs) + " |",
             "|---|" + "---|" * len(runs)]
    for v in VERDICTS:
        lines.append(f"| {v.capitalize()} | " + " | ".join(str(c[r][v]) for r in runs) + " |")
    lines.append("| Unanswerable questions correctly refused | "
                 + " | ".join(f"{c[r]['correctly refused']} of {n_un}" for r in runs) + " |")
    lines += ["", "Answers judged wrong:", ""]
    for r in rows:
        if r["final_verdict"] in ("wrong", "wrongly answered"):
            lines.append(f"- {r['id']} ({r['run']}): {r['note']}")
    (RES / "human_review.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
