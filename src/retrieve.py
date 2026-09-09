"""
Retrieve similar, historically-resolved threads to ground reply drafting. Uses TF-IDF
cosine similarity over resolved threads' customer messages — deliberately simple and
auditable rather than an embedding black box, since "grounded in how this brand has
historically resolved similar issues" needs to be checkable by a human reviewer.

If this proves too weak in eval (see reports/report.md), swap in embeddings — the
interface (`retrieve(query, k)`) doesn't change either way.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THREADS_PATH = Path("data/processed/threads.jsonl")
_cfg = yaml.safe_load(open("config.yaml"))

_index = None  # lazy-built: (vectorizer, matrix, resolved_threads)


def _build_index():
    global _index
    threads = [json.loads(l) for l in open(THREADS_PATH)]
    resolved = [t for t in threads if t.get("resolved")]
    queries = [
        next((x["text"] for x in t["turns"] if x["author"] == "customer"), "")
        for t in resolved
    ]
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vec.fit_transform(queries)
    _index = (vec, matrix, resolved)


def retrieve(message: str, k: int | None = None) -> list[dict]:
    """Returns up to k resolved threads similar to `message`, each annotated with
    `similarity`. Empty list means: no strong precedent found (a meaningful signal,
    not just missing data — see escalate.py)."""
    if _index is None:
        _build_index()
    vec, matrix, resolved = _index
    k = k or _cfg["retrieval"]["top_k"]
    min_sim = _cfg["retrieval"]["min_similarity"]

    query_vec = vec.transform([message])
    sims = cosine_similarity(query_vec, matrix)[0]
    top_idx = np.argsort(sims)[::-1][:k]

    results = []
    for i in top_idx:
        if sims[i] < min_sim:
            continue
        thread = dict(resolved[i])
        thread["similarity"] = float(sims[i])
        results.append(thread)
    return results
