"""Ranking metrics, implemented from their definitions (linear gain for nDCG, as in BEIR).

A *run* is {query id: [(doc id, score), ...]} sorted by decreasing score.
Qrels are {query id: {doc id: relevance > 0}}.
"""
import math

import numpy as np


def ndcg_at_k(ranked, rels, k=10):
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, rels, k=100):
    if not rels:
        return 0.0
    return sum(1 for d in ranked[:k] if d in rels) / len(rels)


def mrr_at_k(ranked, rels, k=10):
    for i, d in enumerate(ranked[:k]):
        if d in rels:
            return 1.0 / (i + 1)
    return 0.0


def ap_at_k(ranked, rels, k=100):
    """Average precision truncated at k, divided by the total number of relevant
    documents (the trec_eval ``map_cut`` definition)."""
    if not rels:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked[:k]):
        if d in rels:
            hits += 1
            total += hits / (i + 1)
    return total / len(rels)


METRICS = {
    "ndcg@10": lambda r, q: ndcg_at_k(r, q, 10),
    "recall@100": lambda r, q: recall_at_k(r, q, 100),
    "mrr@10": lambda r, q: mrr_at_k(r, q, 10),
    "map@100": lambda r, q: ap_at_k(r, q, 100),
}


def evaluate(run, qrels):
    """Return (mean metrics, per-query metrics) over the queries in ``qrels``."""
    per_query = {name: {} for name in METRICS}
    for qid, rels in qrels.items():
        ranked = [d for d, _ in run.get(qid, [])]
        for name, fn in METRICS.items():
            per_query[name][qid] = fn(ranked, rels)
    means = {name: float(np.mean(list(v.values()))) for name, v in per_query.items()}
    return means, per_query
