"""Finding the passages that answer a question.

Three ways, compared in the evaluation:
  * BM25 keyword search (classic search-engine scoring; pure Python, runs anywhere, used online);
  * embedding search (meaning-based: text -> vector with a local embedding model, cosine similarity);
  * hybrid: both rankings combined with reciprocal rank fusion (RRF).
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMB_FILE = ROOT / "data" / "embeddings.npy"
EMB_META = ROOT / "data" / "embeddings.json"

STOP = set("""a an and are as at be been by can could do does for from has have how if in into is it its may
might must not of on or should such than that the their them then there these they this those to under was
were what when where whether which who will with would within any all also other its""".split())
TOKEN = re.compile(r"[a-z0-9]+(?:[/.\-][a-z0-9]+)*")


def tokens(text):
    out = []
    for t in TOKEN.findall(text.lower()):
        if t in STOP:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]                              # light plural folding: devices -> device
        out.append(t)
    return out


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # document number and title are searchable too; detected headings are not used (they are approximate)
        self.docs = [tokens(f"{c['doc_id']} {c['title']} {c['text']}") for c in chunks]
        self.lengths = [len(d) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.docs)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        q = tokens(query)
        out = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg))
            out[i] = s
        return out

    def search(self, query, k=5):
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in order]


class EmbeddingIndex:
    """Cosine similarity over pre-computed chunk vectors. embed_fn turns a query into a vector."""

    def __init__(self, vectors, embed_fn):
        v = np.asarray(vectors, dtype=np.float32)
        self.vectors = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.embed_fn = embed_fn

    def search(self, query, k=5):
        q = np.asarray(self.embed_fn(query), dtype=np.float32)
        q /= np.linalg.norm(q)
        s = self.vectors @ q
        order = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in order]


def rrf(rankings, k=5, c=60):
    """Reciprocal rank fusion: each list adds 1/(c + rank) to a passage's score."""
    score = Counter()
    for ranking in rankings:
        for rank, (i, _) in enumerate(ranking, start=1):
            score[i] += 1.0 / (c + rank)
    return [(i, s) for i, s in score.most_common(k)]


class Hybrid:
    def __init__(self, bm25, emb, depth=30):
        self.bm25, self.emb, self.depth = bm25, emb, depth

    def search(self, query, k=5):
        return rrf([self.bm25.search(query, self.depth), self.emb.search(query, self.depth)], k=k)


def load_embeddings(chunks):
    """Vectors saved by scripts/embed.py; checked against the current chunk list."""
    meta = json.loads(EMB_META.read_text())
    if meta["chunk_ids"] != [c["chunk_id"] for c in chunks]:
        raise ValueError("embeddings.npy is out of date: re-run scripts/embed.py after re-ingesting")
    return np.load(EMB_FILE), meta
