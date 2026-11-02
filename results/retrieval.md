## Retrieval: is a correct passage (right document and page) found?

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@8 | Right document in top 5 | MRR@10 |
|---|---|---|---|---|---|---|
| BM25 (keywords) | 65% | 78% | 88% | 98% | 98% | 0.73 |
| Embeddings (nomic-embed-text) | 60% | 85% | 90% | 95% | 95% | 0.72 |
| Hybrid (BM25 + embeddings) | 68% | 85% | 92% | 98% | 95% | 0.78 |
