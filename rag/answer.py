"""Answer a question from retrieved passages, with citations, or say the guidance does not cover it."""
from . import ollama

SYSTEM = """You answer questions about EU medical device regulation using ONLY the numbered passages from
MDCG guidance documents given in the user message.

Rules:
- Use only facts stated in the passages. Do not use outside knowledge, even if you know the answer.
- Cite the passages you used by their numbers in "citations".
- If the passages do not contain the answer, set "covered" to false, give an empty citation list and
  answer exactly: "The provided MDCG guidance does not cover this."
- Keep the answer short (2-4 sentences), precise, and in plain English. Keep numbers, deadlines and
  device classes exactly as written in the passages."""

SCHEMA = {
    "type": "object",
    "properties": {
        "covered": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["covered", "answer", "citations"],
}
NOT_COVERED = "The provided MDCG guidance does not cover this."


def format_passages(chunks):
    return "\n\n".join(f"[{n}] {c['doc_id']} ({c['title']}), page {c['page_start']}"
                       f"{'' if c['page_end'] == c['page_start'] else '-' + str(c['page_end'])}:\n{c['text']}"
                       for n, c in enumerate(chunks, start=1))


def answer(question, chunks, chat=ollama.chat_json, **kw):
    """chunks: the retrieved passages, best first. Returns a dict with answer, sources and usage."""
    user = f"Passages:\n\n{format_passages(chunks)}\n\nQuestion: {question}"
    reply, usage = chat(SYSTEM, user, SCHEMA, **kw)
    cited = [i for i in dict.fromkeys(reply.get("citations") or []) if isinstance(i, int) and 1 <= i <= len(chunks)]
    covered = bool(reply.get("covered")) and bool(cited)
    return {"question": question, "covered": covered,
            "answer": reply.get("answer", "").strip() if covered else NOT_COVERED,
            "sources": [chunks[i - 1] for i in cited] if covered else [],
            "invalid_citations": [i for i in (reply.get("citations") or []) if i not in cited],
            "usage": usage}
