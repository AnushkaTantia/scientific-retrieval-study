"""Run a retriever over a dataset and record accuracy, latency and index size."""
import json
import time
from pathlib import Path

import numpy as np

from .metrics import evaluate


def run_retriever(retriever, ds, k=100, warmup=5, fit_kwargs=None):
    """Fit ``retriever`` on ``ds`` and run every query one at a time.

    Latency is wall-clock time per single query (query processing + scoring + top-k),
    measured on one thread after a short warm-up.
    """
    t0 = time.perf_counter()
    if fit_kwargs:  # e.g. precomputed hashed features
        retriever.fit(**fit_kwargs)
    else:
        retriever.fit(ds.doc_texts)
    build_seconds = time.perf_counter() - t0

    for q in ds.query_texts[:warmup]:
        retriever.search(q, k)

    run, times = {}, []
    for qid, q in zip(ds.query_ids, ds.query_texts):
        t = time.perf_counter()
        idx, scores = retriever.search(q, k)
        times.append((time.perf_counter() - t) * 1000.0)
        run[qid] = [(ds.doc_ids[i], float(s)) for i, s in zip(idx, scores)]

    means, per_query = evaluate(run, ds.qrels)
    return {
        "dataset": ds.name,
        "method": retriever.name,
        "n_docs": len(ds.doc_ids),
        "n_queries": len(ds.query_ids),
        "metrics": means,
        "per_query": per_query,
        "build_seconds": build_seconds,
        "latency_ms": {"mean": float(np.mean(times)), "p50": float(np.percentile(times, 50)),
                       "p95": float(np.percentile(times, 95))},
        "index_mb": retriever.index_bytes() / 1e6,
    }, run


def save_result(result, run, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{result['dataset']}__{result['method']}"
    (out / f"{stem}.json").write_text(json.dumps(result, indent=1))
    (out / f"{stem}.run.json").write_text(json.dumps(run))
