"""Write failed queries of the best system (nDCG@10 = 0) for manual error analysis.

    python scripts/dump_failures.py --laptop laptop --per-dataset 15
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irstudy.data import load_beir
from irstudy.fusion import weighted_fusion
from irstudy.sparse import BM25

ap = argparse.ArgumentParser()
ap.add_argument("--laptop", default=".")
ap.add_argument("--per-dataset", type=int, default=15)
ap.add_argument("--out", default="results_final")
args = ap.parse_args()
sweep = pd.read_csv(Path(args.out) / "fusion_alpha_sweep.csv")
pq = pd.read_csv(Path(args.out) / "per_query.csv", dtype={"qid": str})
rows = []
for name in ["scifact", "nfcorpus"]:
    ds = load_beir("data", name)
    alpha = float(sweep[(sweep.dataset == name) & (sweep.dense == "bge-small") & sweep.chosen].alpha.iloc[0])
    bm = BM25(stem=True).fit(ds.doc_texts)
    c = Path(args.laptop) / "cache" / name
    E = np.load(c / "bge-small__docs.npy"); Q = np.load(c / "bge-small__queries_test.npy")
    sub = pq[(pq.dataset == name) & (pq["hybrid-wsum-bge-small"] == 0)]
    print(name, "failed queries:", len(sub), "of", (pq.dataset == name).sum())
    for _, r in sub.sample(n=min(args.per_dataset, len(sub)), random_state=0).iterrows():
        i = ds.query_ids.index(r.qid)
        order = list(np.argsort(-weighted_fusion(bm.scores(ds.query_texts[i]), E @ Q[i], alpha), kind="stable"))
        rel = ds.qrels[r.qid]
        first_rel = min(order.index(ds.doc_ids.index(d)) for d in rel) + 1
        rows.append({"dataset": name, "qid": r.qid, "query": ds.query_texts[i], "n_relevant": len(rel),
                     "first_relevant_rank": first_rel,
                     "relevant": " || ".join(ds.doc_texts[ds.doc_ids.index(d)][:220] for d in list(rel)[:2]),
                     "retrieved_top3": " || ".join(ds.doc_texts[j][:160] for j in order[:3])})
pd.DataFrame(rows).to_csv(Path(args.out) / "failures_sample.csv", index=False)
