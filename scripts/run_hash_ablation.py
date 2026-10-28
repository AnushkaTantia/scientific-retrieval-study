"""Hashing ablations: number of buckets, n-gram features, and signed vs unsigned hashing.

    python scripts/run_hash_ablation.py --out results_ablation

Accuracy and index size are deterministic. Latencies here come from whichever machine runs
this script, so compare them only with each other, not with the main latency table.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irstudy.data import load_beir  # noqa: E402
from irstudy.evaluate import run_retriever  # noqa: E402
from irstudy.hashed import HashedRetriever, hash32  # noqa: E402

FEATURE_CONFIGS = {
    "word 1": dict(word_ngrams=(1, 1), char_ngrams=None),
    "word 1-2": dict(word_ngrams=(1, 2), char_ngrams=None),
    "word 1-3": dict(word_ngrams=(1, 3), char_ngrams=None),
    "char 3-5": dict(word_ngrams=None, char_ngrams=(3, 5)),
    "word 1 + char 3-4": dict(word_ngrams=(1, 1), char_ngrams=(3, 4)),
    "word 1 + char 3-5": dict(word_ngrams=(1, 1), char_ngrams=(3, 5)),
}
BITS = [10, 12, 14, 16, 18]
SIGN_CONFIGS = ["word 1-2", "word 1 + char 3-5"]


def collision_stats(vocab, bits):
    """Share of distinct features that land in a bucket shared with another feature."""
    buckets = np.array([hash32(f) & ((1 << bits) - 1) for f in vocab])
    _, inverse, counts = np.unique(buckets, return_inverse=True, return_counts=True)
    return float((counts[inverse] > 1).mean()), int(len(counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results_ablation")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "hash_ablation.csv"
    done = set()
    if csv_path.exists():  # resume: skip settings already recorded
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["features"], int(r["bits"]), r["signed"] == "True"))
    writer, fh = None, None
    for name in args.datasets:
        ds = load_beir(args.data_dir, name)
        print(ds.summary(), flush=True)
        for label, cfg in FEATURE_CONFIGS.items():
            todo = [(b, s) for b in BITS for s in ([True, False] if label in SIGN_CONFIGS else [True])
                    if (name, label, b, s) not in done]
            if not todo:
                continue
            t0 = time.perf_counter()
            counters = HashedRetriever(**cfg).extract(ds.doc_texts)
            extract_s = time.perf_counter() - t0
            vocab = set().union(*counters)
            for bits, signed in todo:
                m = HashedRetriever(bits=bits, signed=signed, **cfg)
                res, _ = run_retriever(m, ds, fit_kwargs={"counters": counters})
                coll, used = collision_stats(vocab, bits)
                row = {"dataset": name, "features": label, "bits": bits, "buckets": 1 << bits,
                       "signed": signed, "distinct_features": len(vocab),
                       "collision_rate": coll, "buckets_used": used,
                       **res["metrics"], "index_mb": res["index_mb"],
                       "latency_ms_mean": res["latency_ms"]["mean"],
                       "build_s": res["build_seconds"] + extract_s}
                if writer is None:
                    new = not csv_path.exists()
                    fh = open(csv_path, "a", newline="")
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    if new:
                        writer.writeheader()
                writer.writerow(row)
                fh.flush()
                print(f"  {label:<18} 2^{bits} signed={signed!s:<5} nDCG@10={row['ndcg@10']:.3f} "
                      f"R@100={row['recall@100']:.3f} idx={row['index_mb']:.1f}MB "
                      f"collisions={coll:.1%}", flush=True)
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
