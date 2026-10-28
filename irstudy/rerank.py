"""Cross-encoder scoring of (query, document) candidate pairs."""
import time

import numpy as np

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_cross_encoder(hf_id=CROSS_ENCODER):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(hf_id, device="cpu", max_length=512)


def score_candidates(model, query, doc_texts, batch_size=32):
    """Return (scores, seconds) for one query against its candidate documents."""
    t0 = time.perf_counter()
    scores = model.predict([(query, d) for d in doc_texts], batch_size=batch_size,
                           show_progress_bar=False)
    return np.asarray(scores, dtype=np.float64), time.perf_counter() - t0


def rerank(first_stage_ids, ce_scores, depth):
    """Re-order the top ``depth`` of a first-stage ranking by cross-encoder score and keep
    the rest of the ranking below them unchanged.

    first_stage_ids: ranked doc ids; ce_scores: {doc id: score} covering the top ``depth``.
    """
    head = sorted(first_stage_ids[:depth], key=lambda d: -ce_scores[d])
    return head + list(first_stage_ids[depth:])
