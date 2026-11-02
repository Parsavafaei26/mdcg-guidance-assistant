"""Turn every passage into a vector with the local embedding model (needs Ollama running).

    python3 scripts/embed.py            # writes data/embeddings.npy and data/embeddings.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from rag import ollama  # noqa: E402
from rag.ingest import load_chunks  # noqa: E402
from rag.retrieve import EMB_FILE, EMB_META  # noqa: E402


def main(batch=16):
    chunks = load_chunks()
    vecs, t0 = [], time.perf_counter()
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        vecs += ollama.embed([f"{c['doc_id']} {c['title']}. {c['text']}" for c in part], kind="document")
        print(f"  {min(i + batch, len(chunks))}/{len(chunks)} passages", flush=True)
    np.save(EMB_FILE, np.asarray(vecs, dtype=np.float32))
    EMB_META.write_text(json.dumps({"model": ollama.EMBED_MODEL, "dim": len(vecs[0]),
                                    "chunk_ids": [c["chunk_id"] for c in chunks]}))
    print(f"Saved {len(vecs)} vectors of size {len(vecs[0])} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
