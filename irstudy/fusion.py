"""Hybrid score fusion over full per-query score vectors (one score per document)."""
import numpy as np


def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def weighted_fusion(lexical, dense, alpha):
    """alpha * dense + (1 - alpha) * lexical, after min-max normalising each per query.
    alpha = 0 is pure lexical, alpha = 1 is pure dense."""
    return alpha * minmax(dense) + (1.0 - alpha) * minmax(lexical)


def ranks_from_scores(scores, depth):
    """1-based rank of every document, with documents outside the top ``depth`` given
    rank infinity (they contribute nothing to RRF)."""
    order = np.argsort(-scores, kind="stable")[:depth]
    r = np.full(len(scores), np.inf)
    r[order] = np.arange(1, len(order) + 1)
    return r


def rrf(score_lists, k=60, depth=1000):
    """Reciprocal Rank Fusion (Cormack et al., 2009): sum over systems of 1 / (k + rank)."""
    total = np.zeros(len(score_lists[0]))
    for s in score_lists:
        total += 1.0 / (k + ranks_from_scores(s, depth))
    return total
