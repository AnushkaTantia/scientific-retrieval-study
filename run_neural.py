"""Model-dependent experiments, run on the laptop (needs sentence-transformers and faiss).

    python run_neural.py

For each dataset this script:
  1. re-runs the lexical methods, so that every latency in the study is measured on the
     same machine (accuracy should match the earlier run exactly);
  2. runs both dense retrievers with an exact FAISS index, and caches document and query
     embeddings (test split and tuning split) for the fusion analysis;
  3. scores the top-100 candidates of BM25 and of a fixed RRF hybrid with a cross-encoder,
     and saves those scores for the re-ranking analysis.
Finished steps are skipped on a re-run, so it is safe to stop and restart.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402

from irstudy.data import load_beir  # noqa: E402
from irstudy.dense import MODELS, DenseRetriever  # noqa: E402
from irstudy.evaluate import run_retriever, save_result  # noqa: E402
from irstudy.fusion import rrf  # noqa: E402
from irstudy.hashed import HashedRetriever  # noqa: E402
from irstudy.rerank import CROSS_ENCODER, load_cross_encoder, score_candidates  # noqa: E402
from irstudy.sparse import BM25, TfidfCosine  # noqa: E402

TUNE_SPLIT = {"scifact": "train", "nfcorpus": "dev"}
CE_DEPTH = 100


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def machine_info(out):
    info = {"platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(), "cpu_count": os.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    (out / "machine.json").write_text(json.dumps(info, indent=1))
    log(f"machine: {info}")


def run_and_save(method, ds, out, fit_kwargs=None):
    if (out / f"{ds.name}__{method.name}.json").exists():
        log(f"  skip {method.name} (done)")
        return
    result, run = run_retriever(method, ds, fit_kwargs=fit_kwargs)
    save_result(result, run, out)
    m = result["metrics"]
    log(f"  {method.name:<24} nDCG@10={m['ndcg@10']:.3f} R@100={m['recall@100']:.3f} "
        f"lat={result['latency_ms']['mean']:.1f}ms idx={result['index_mb']:.1f}MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results_laptop")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--skip-rerank", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    machine_info(out)

    for name in args.datasets:
        ds = load_beir(args.data_dir, name)
        tune = load_beir(args.data_dir, name, TUNE_SPLIT[name])
        cache = Path(args.cache) / name
        log(ds.summary())

        # 1. lexical methods, re-timed on this machine
        for method in [TfidfCosine(), BM25(), BM25(stem=True),
                       HashedRetriever(bits=16, word_ngrams=(1, 2)),
                       HashedRetriever(bits=16, word_ngrams=(1, 1), char_ngrams=(3, 5))]:
            run_and_save(method, ds, out)

        # 2. dense retrievers + cached query embeddings
        dense = {}
        for short in MODELS:
            log(f"  dense model {short}: encoding corpus (first run takes a few minutes)")
            d = DenseRetriever(short, cache)
            run_and_save(d, ds, out, fit_kwargs={"doc_texts": ds.doc_texts, "doc_ids": ds.doc_ids})
            if not hasattr(d, "doc_emb"):
                d.fit(ds.doc_texts, ds.doc_ids)
            for split, qds in [("test", ds), (TUNE_SPLIT[name], tune)]:
                path = cache / f"{short}__queries_{split}.npy"
                if not path.exists():
                    np.save(path, d.encode_queries(qds.query_texts))
                    (cache / f"{short}__query_ids_{split}.json").write_text(json.dumps(qds.query_ids))
            dense[short] = d

        if args.skip_rerank:
            continue

        # 3. cross-encoder scores for the top-100 of BM25-stem and of RRF(BM25-stem, bge-small)
        ce_path = out / f"{name}__ce_scores.json"
        if ce_path.exists():
            log("  skip cross-encoder (done)")
            continue
        log(f"  cross-encoder {CROSS_ENCODER}: scoring candidates (this is the slow step)")
        ce = load_cross_encoder()
        bm25 = BM25(stem=True).fit(ds.doc_texts)
        q_emb = np.load(cache / "bge-small__queries_test.npy")
        doc_emb = dense["bge-small"].doc_emb
        scores, timing = {}, {}
        t_start = time.perf_counter()
        for i, (qid, q) in enumerate(zip(ds.query_ids, ds.query_texts)):
            lex = bm25.scores(q)
            hyb = rrf([lex, doc_emb @ q_emb[i]])
            bm25_top = list(np.argsort(-lex, kind="stable")[:CE_DEPTH])
            seen = set(int(j) for j in bm25_top)
            hyb_top = [j for j in np.argsort(-hyb, kind="stable")[:CE_DEPTH] if int(j) not in seen]
            s1, t1 = score_candidates(ce, q, [ds.doc_texts[j] for j in bm25_top])
            s2, t2 = (score_candidates(ce, q, [ds.doc_texts[j] for j in hyb_top])
                      if hyb_top else (np.zeros(0), 0.0))
            scores[qid] = {ds.doc_ids[j]: float(s) for j, s in
                           zip(bm25_top + hyb_top, np.concatenate([s1, s2]))}
            timing[qid] = {"bm25_top100_seconds": t1, "extra_pairs": len(hyb_top),
                           "extra_seconds": t2}
            if (i + 1) % 25 == 0:
                done = time.perf_counter() - t_start
                log(f"    {i + 1}/{len(ds.query_ids)} queries, "
                    f"~{done / (i + 1) * (len(ds.query_ids) - i - 1) / 60:.0f} min left")
        ce_path.write_text(json.dumps({"model": CROSS_ENCODER, "depth": CE_DEPTH,
                                       "scores": scores, "timing": timing}))
        log("  cross-encoder done")
    log("ALL DONE")


if __name__ == "__main__":
    main()
