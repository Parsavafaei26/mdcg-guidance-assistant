"""Minimal client for a local Ollama server (standard library only).

Ollama runs open-source models on this computer: nothing is sent to an outside service and nothing is paid.
"""
import json
import os
import time
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
CHAT_MODEL = os.environ.get("OLLAMA_MODEL", "granite4.2:8b")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


class OllamaError(RuntimeError):
    pass


def _post(path, body, opener=urllib.request.urlopen, timeout=300):
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json"})
    try:
        with opener(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise OllamaError(f"Ollama error {exc.code}: {exc.read()[:300]!r}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise OllamaError(f"Cannot reach Ollama at {HOST}: is the Ollama app running? ({exc})") from None


def available(opener=urllib.request.urlopen):
    try:
        with opener(HOST + "/api/tags", timeout=3) as resp:
            return [m["name"] for m in json.loads(resp.read()).get("models", [])]
    except Exception:
        return None


def embed(texts, model=EMBED_MODEL, kind="query", opener=urllib.request.urlopen):
    """Vectors for a list of texts. nomic-embed-text expects a task prefix on every text."""
    prefix = {"query": "search_query: ", "document": "search_document: "}[kind] if "nomic" in model else ""
    out = _post("/api/embed", {"model": model, "input": [prefix + t for t in texts]}, opener)
    return out["embeddings"]


def chat_json(system, user, schema, model=CHAT_MODEL, opener=urllib.request.urlopen, num_predict=600):
    """One chat turn whose reply is forced into `schema`. Returns (parsed JSON, usage dict)."""
    body = {"model": model, "stream": False, "format": schema, "think": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": num_predict}}
    t0 = time.perf_counter()
    try:
        out = _post("/api/chat", body, opener)
    except OllamaError as exc:
        if "think" not in str(exc).lower():
            raise
        body.pop("think")                       # model without a thinking switch
        out = _post("/api/chat", body, opener)
    usage = {"prompt_tokens": out.get("prompt_eval_count", 0), "output_tokens": out.get("eval_count", 0),
             "seconds": round(time.perf_counter() - t0, 2), "done_reason": out.get("done_reason")}
    if out.get("done_reason") == "length":
        raise OllamaError(f"reply cut off at {num_predict} tokens")
    try:
        return json.loads(out["message"]["content"]), usage
    except json.JSONDecodeError:
        raise OllamaError(f"reply was not valid JSON: {out['message']['content'][:200]!r}") from None
