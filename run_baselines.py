"""Day 1-2: lexical baselines and the hashed retriever.

    python run_baselines.py --datasets scifact nfcorpus --data-dir data --out results
"""
import argparse
import csv
from pathlib import Path

from irstudy.data import load_beir
from irstudy.evaluate import run_retriever, save_result
from irstudy.hashed import HashedRetriever
from irstudy.sparse import BM25, TfidfCosine


def build_methods():
    return [
        TfidfCosine(),
        BM25(),
        BM25(stem=True),
        HashedRetriever(bits=16, word_ngrams=(1, 2)),
        HashedRetriever(bits=16, word_ngrams=(1, 1), char_ngrams=(3, 5)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows = []
    for name in args.datasets:
        ds = load_beir(args.data_dir, name)
        print(ds.summary())
        for method in build_methods():
            result, run = run_retriever(method, ds)
            save_result(result, run, args.out)
            m = result["metrics"]
            print(f"  {result['method']:<28} nDCG@10={m['ndcg@10']:.3f} R@100={m['recall@100']:.3f} "
                  f"MRR@10={m['mrr@10']:.3f} MAP@100={m['map@100']:.3f} "
                  f"lat={result['latency_ms']['mean']:.1f}ms idx={result['index_mb']:.1f}MB")
            rows.append({"dataset": name, "method": result["method"], **m,
                         "latency_ms_mean": result["latency_ms"]["mean"],
                         "latency_ms_p95": result["latency_ms"]["p95"],
                         "index_mb": result["index_mb"], "build_s": result["build_seconds"]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary_baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
