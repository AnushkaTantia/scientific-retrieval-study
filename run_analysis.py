"""Day 4-5: fusion, re-ranking, significance tests and per-query analysis.

    python run_analysis.py --laptop laptop --out results_final

Needs: results/ (lexical runs from run_baselines.py), and the laptop outputs of
run_neural.py (cache/ with embeddings, results_laptop/ with latencies and cross-encoder
scores). All accuracy numbers are computed here from those files; latencies all come from
the laptop, so they are measured on one machine.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from irstudy.data import load_beir
from irstudy.fusion import rrf, weighted_fusion
from irstudy.metrics import evaluate
from irstudy.rerank import rerank
from irstudy.sparse import BM25, top_k
from irstudy.stats import paired_bootstrap

TUNE_SPLIT = {"scifact": "train", "nfcorpus": "dev"}
DENSE = {"minilm": "MiniLM-L6", "bge-small": "bge-small"}
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)
DEPTHS = [10, 20, 30, 50, 100]
LEXICAL = {"tfidf": "TF-IDF", "bm25": "BM25", "bm25-stem": "BM25 + stemming",
           "hashed-w1-2-b16": "Hashed, word 1-2 (2^16)",
           "hashed-w1-1+c3-5-b16": "Hashed, word 1 + char 3-5 (2^16)"}


def as_run(ds, ranked_idx):
    return {q: [(ds.doc_ids[i], 0.0) for i in r] for q, r in zip(ds.query_ids, ranked_idx)}


def rank_matrix(S, k=100):
    return [top_k(row, k)[0] for row in S]


def ndcg_mean(ds, S):
    return evaluate(as_run(ds, rank_matrix(S, 10)), ds.qrels)[0]["ndcg@10"]


def fuse_rows(L, D, alpha):
    """Weighted fusion applied query by query, so min-max normalisation is per query."""
    return np.vstack([weighted_fusion(l, d, alpha) for l, d in zip(L, D)])


def holm(pvals):
    order = np.argsort(pvals)
    m, adj, running = len(pvals), np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def analyse(name, args, out):
    lap = Path(args.laptop)
    ds = load_beir(args.data_dir, name)
    tune = load_beir(args.data_dir, name, TUNE_SPLIT[name])
    lat = {}  # laptop latency per query, ms
    size = {}  # index size, MB

    bm25 = BM25(stem=True).fit(ds.doc_texts)
    L = np.vstack([bm25.scores(q) for q in ds.query_texts])
    L_tune = np.vstack([bm25.scores(q) for q in tune.query_texts])

    runs, labels, family = {}, {}, {}
    # lexical and hashed: runs from run_baselines.py, latency from the laptop
    for key, label in LEXICAL.items():
        r = json.loads((Path(args.results) / f"{name}__{key}.run.json").read_text())
        runs[key] = {q: [d for d, _ in r[q]] for q in ds.query_ids}
        info = json.loads((lap / "results_laptop" / f"{name}__{key}.json").read_text())
        lat[key], size[key] = info["latency_ms"]["mean"], info["index_mb"]
        labels[key], family[key] = label, "hashed" if key.startswith("hashed") else "lexical"

    # dense
    D, D_tune = {}, {}
    for m, label in DENSE.items():
        c = lap / "cache" / name
        E = np.load(c / f"{m}__docs.npy")
        assert json.loads((c / f"{m}__doc_ids.json").read_text()) == ds.doc_ids
        assert json.loads((c / f"{m}__query_ids_test.json").read_text()) == ds.query_ids
        assert json.loads((c / f"{m}__query_ids_{TUNE_SPLIT[name]}.json").read_text()) == tune.query_ids
        # one query at a time (E @ q), exactly as run_neural.py computed the candidates that
        # were sent to the cross-encoder; a single matrix product differs in the last bits
        # of float32 and can reorder near-tied documents
        D[m] = np.vstack([E @ q for q in np.load(c / f"{m}__queries_test.npy")])
        D_tune[m] = np.vstack([E @ q for q in np.load(c / f"{m}__queries_{TUNE_SPLIT[name]}.npy")])
        key = f"dense-{m}"
        runs[key] = {q: [ds.doc_ids[i] for i in r] for q, r in zip(ds.query_ids, rank_matrix(D[m]))}
        info = json.loads((lap / "results_laptop" / f"{name}__{key}.json").read_text())
        lat[key], size[key] = info["latency_ms"]["mean"], E.nbytes / 1e6
        labels[key], family[key] = label, "dense"

    # hybrids: alpha chosen on the tuning split only, then applied once to the test split
    sweep = []
    for m, label in DENSE.items():
        tune_scores = [ndcg_mean(tune, fuse_rows(L_tune, D_tune[m], a)) for a in ALPHAS]
        test_scores = [ndcg_mean(ds, fuse_rows(L, D[m], a)) for a in ALPHAS]
        best = float(ALPHAS[int(np.argmax(tune_scores))])
        sweep += [{"dataset": name, "dense": label, "alpha": float(a), "ndcg@10_tune": t,
                   "ndcg@10_test": s, "chosen": float(a) == best}
                  for a, t, s in zip(ALPHAS, tune_scores, test_scores)]
        key = f"hybrid-wsum-{m}"
        H = fuse_rows(L, D[m], best)
        runs[key] = {q: [ds.doc_ids[i] for i in r] for q, r in zip(ds.query_ids, rank_matrix(H))}
        labels[key], family[key] = f"Weighted fusion, BM25-stem + {label} (alpha={best:.2f})", "hybrid"
        key = f"hybrid-rrf-{m}"
        R = np.vstack([rrf([L[i], D[m][i]]) for i in range(len(L))])
        runs[key] = {q: [ds.doc_ids[i] for i in r] for q, r in zip(ds.query_ids, rank_matrix(R))}
        labels[key], family[key] = f"RRF, BM25-stem + {label}", "hybrid"
        for k in (f"hybrid-wsum-{m}", f"hybrid-rrf-{m}"):
            # both components run per query; fusion itself costs well under a millisecond
            lat[k], size[k] = lat["bm25-stem"] + lat[f"dense-{m}"], size["bm25-stem"] + size[f"dense-{m}"]

    # cross-encoder re-ranking of two first stages, at several depths
    ce = json.loads((lap / "results_laptop" / f"{name}__ce_scores.json").read_text())
    t = ce["timing"].values()
    ms_per_pair = 1000 * sum(v["bm25_top100_seconds"] + v["extra_seconds"] for v in t) / \
        sum(100 + v["extra_pairs"] for v in t)
    first_stages = {"bm25-stem": [list(top_k(L[i], 100, drop_zero=True)[0]) for i in range(len(L))],
                    "hybrid-rrf-bge-small": [list(np.argsort(-rrf([L[i], D["bge-small"][i]]),
                                                             kind="stable")[:100]) for i in range(len(L))]}
    depth_rows = []
    for fs, ranked in first_stages.items():
        for depth in DEPTHS:
            run = {}
            for q, r in zip(ds.query_ids, ranked):
                ids = [ds.doc_ids[i] for i in r]
                missing = [d for d in ids[:depth] if d not in ce["scores"][q]]
                assert not missing, f"no cross-encoder score for {missing[:3]}"
                run[q] = rerank(ids, ce["scores"][q], depth)
            key = f"ce{depth}-{fs}"
            runs[key] = run
            labels[key] = f"{labels[fs]} + cross-encoder@{depth}"
            family[key] = "re-ranked"
            lat[key] = lat[fs] + depth * ms_per_pair
            size[key] = size[fs]
            m_all = evaluate({q: [(d, 0) for d in r] for q, r in run.items()}, ds.qrels)[0]
            depth_rows.append({"dataset": name, "first_stage": labels[fs], "depth": depth,
                               **m_all, "latency_ms": lat[key]})

    # evaluate everything
    per_query, rows = {}, []
    for key, run in runs.items():
        means, pq = evaluate({q: [(d, 0) for d in r] for q, r in run.items()}, ds.qrels)
        per_query[key] = pq
        if key.startswith("ce") and not key.startswith("ce100"):
            continue  # other depths are reported in the depth table
        rows.append({"dataset": name, "key": key, "method": labels[key], "family": family[key],
                     **means, "latency_ms": lat[key], "index_mb": size[key]})
    main = pd.DataFrame(rows)

    # significance on nDCG@10 (paired bootstrap, Holm-corrected within the dataset)
    pairs = [("dense-bge-small", "bm25-stem"), ("dense-minilm", "bm25-stem"),
             ("hashed-w1-1+c3-5-b16", "bm25-stem"), ("hashed-w1-1+c3-5-b16", "bm25"),
             ("hashed-w1-1+c3-5-b16", "dense-minilm"),
             ("hybrid-wsum-bge-small", "dense-bge-small"), ("hybrid-rrf-bge-small", "dense-bge-small"),
             ("hybrid-wsum-minilm", "dense-minilm"),
             ("ce100-bm25-stem", "bm25-stem"), ("ce100-hybrid-rrf-bge-small", "hybrid-rrf-bge-small"),
             ("ce100-hybrid-rrf-bge-small", "dense-bge-small")]
    sig = []
    for a, b in pairs:
        qs = ds.query_ids
        res = paired_bootstrap([per_query[a]["ndcg@10"][q] for q in qs],
                               [per_query[b]["ndcg@10"][q] for q in qs])
        wins = sum(per_query[a]["ndcg@10"][q] > per_query[b]["ndcg@10"][q] for q in qs)
        losses = sum(per_query[a]["ndcg@10"][q] < per_query[b]["ndcg@10"][q] for q in qs)
        sig.append({"dataset": name, "system_a": labels[a], "system_b": labels[b],
                    "ndcg@10_a": np.mean([per_query[a]["ndcg@10"][q] for q in qs]),
                    "ndcg@10_b": np.mean([per_query[b]["ndcg@10"][q] for q in qs]),
                    "diff": res["mean_diff"], "ci95_low": res["ci95"][0], "ci95_high": res["ci95"][1],
                    "p_bootstrap": res["p_bootstrap"], "p_ttest": res["p_ttest"],
                    "wins": wins, "ties": len(qs) - wins - losses, "losses": losses})
    sig = pd.DataFrame(sig)
    sig["p_holm"] = holm(sig["p_bootstrap"].to_numpy())
    sig["significant_0.05"] = sig["p_holm"] < 0.05

    # per-query table for length and error analysis
    qlen = {q: len(t.split()) for q, t in zip(ds.query_ids, ds.query_texts)}
    pq_rows = []
    for q, text in zip(ds.query_ids, ds.query_texts):
        row = {"dataset": name, "qid": q, "query": text, "query_words": qlen[q],
               "n_relevant": len(ds.qrels[q])}
        for key in ["bm25-stem", "hashed-w1-1+c3-5-b16", "dense-bge-small", "hybrid-wsum-bge-small",
                    "hybrid-rrf-bge-small", "ce100-hybrid-rrf-bge-small"]:
            row[key] = per_query[key]["ndcg@10"][q]
        row["best_top3"] = " || ".join(ds.doc_texts[ds.doc_ids.index(d)][:90]
                                       for d in runs["ce100-hybrid-rrf-bge-small"][q][:3])
        row["relevant_docs"] = " || ".join(ds.doc_texts[ds.doc_ids.index(d)][:90]
                                           for d in list(ds.qrels[q])[:3])
        pq_rows.append(row)

    stats = {"dataset": name, "documents": len(ds.doc_ids), "test_queries": len(ds.query_ids),
             "tuning_queries": len(tune.query_ids), "tuning_split": TUNE_SPLIT[name],
             "relevant_per_query": float(np.mean([len(v) for v in ds.qrels.values()])),
             "query_words_mean": float(np.mean(list(qlen.values()))),
             "doc_words_mean": float(np.mean([len(t.split()) for t in ds.doc_texts])),
             "ce_ms_per_pair": ms_per_pair}
    return main, pd.DataFrame(sweep), pd.DataFrame(depth_rows), sig, pd.DataFrame(pq_rows), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--laptop", default=".")
    ap.add_argument("--out", default="results_final")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = [analyse(n, args, out) for n in args.datasets]
    names = ["main_table", "fusion_alpha_sweep", "rerank_depth", "significance", "per_query"]
    for i, n in enumerate(names):
        pd.concat([p[i] for p in parts]).to_csv(out / f"{n}.csv", index=False)
    (out / "dataset_stats.json").write_text(json.dumps([p[5] for p in parts], indent=1))
    print(pd.concat([p[0] for p in parts]).drop(columns=["key"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
