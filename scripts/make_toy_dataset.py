"""Generate a small SYNTHETIC dataset in BEIR format, for smoke tests only.

The text is random topic vocabulary, so results on it say nothing about retrieval quality
on real data and must never be reported. It exists to check that the code runs end to end.
"""
import json
import random
import sys
from pathlib import Path


def make(out_dir, n_topics=12, docs_per_topic=25, n_queries=40, seed=0):
    rng = random.Random(seed)
    topics = [[f"t{t}w{i}" for i in range(40)] for t in range(n_topics)]
    common = [f"common{i}" for i in range(60)]
    root = Path(out_dir) / "toy"
    (root / "qrels").mkdir(parents=True, exist_ok=True)

    docs = {}
    for t, vocab in enumerate(topics):
        for j in range(docs_per_topic):
            words = [rng.choice(vocab) if rng.random() < 0.6 else rng.choice(common)
                     for _ in range(80)]
            docs[f"d{t}_{j}"] = (t, " ".join(words[:8]), " ".join(words[8:]))

    with open(root / "corpus.jsonl", "w") as f:
        for did, (_, title, text) in docs.items():
            f.write(json.dumps({"_id": did, "title": title, "text": text}) + "\n")

    queries, qrels = [], []
    for q in range(n_queries):
        t = q % n_topics
        words = rng.sample(topics[t], 5) + rng.sample(common, 2)
        queries.append({"_id": f"q{q}", "text": " ".join(words)})
        relevant = rng.sample([d for d, v in docs.items() if v[0] == t], 6)
        for r, did in enumerate(relevant):
            qrels.append((f"q{q}", did, 2 if r < 2 else 1))
    with open(root / "queries.jsonl", "w") as f:
        for row in queries:
            f.write(json.dumps(row) + "\n")
    with open(root / "qrels" / "test.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q, d, s in qrels:
            f.write(f"{q}\t{d}\t{s}\n")
    return root


if __name__ == "__main__":
    print(make(sys.argv[1] if len(sys.argv) > 1 else "data"))
