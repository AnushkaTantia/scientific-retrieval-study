import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from irstudy.data import load_beir  # noqa: E402
from irstudy.evaluate import run_retriever  # noqa: E402
from irstudy.hashed import HashedRetriever  # noqa: E402
from irstudy.metrics import ap_at_k, mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402
from irstudy.sparse import BM25, TfidfCosine, top_k  # noqa: E402
from irstudy.stats import paired_bootstrap  # noqa: E402
from irstudy.text import tokenize  # noqa: E402
from make_toy_dataset import make  # noqa: E402


# ---- metrics, checked by hand -------------------------------------------
def test_metrics_by_hand():
    ranked = ["a", "b", "c", "d"]
    rels = {"b": 1, "d": 2, "z": 1}
    dcg = 1 / math.log2(3) + 2 / math.log2(5)
    idcg = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranked, rels, 10) == pytest.approx(dcg / idcg)
    assert recall_at_k(ranked, rels, 100) == pytest.approx(2 / 3)
    assert mrr_at_k(ranked, rels, 10) == pytest.approx(1 / 2)
    assert mrr_at_k(ranked, rels, 1) == 0.0
    assert ap_at_k(ranked, rels, 100) == pytest.approx((1 / 2 + 2 / 4) / 3)


def test_ndcg_matches_sklearn():
    from sklearn.metrics import ndcg_score
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = 30
        gains = rng.integers(0, 3, size=n)
        if gains.sum() == 0:
            continue
        scores = rng.permutation(n).astype(float)  # no ties
        order = np.argsort(-scores)
        ranked = [str(i) for i in order]
        rels = {str(i): int(g) for i, g in enumerate(gains) if g > 0}
        assert ndcg_at_k(ranked, rels, 10) == pytest.approx(
            ndcg_score([gains], [scores], k=10), abs=1e-9)


def test_perfect_and_empty_rankings():
    rels = {"a": 1, "b": 2}
    assert ndcg_at_k(["b", "a", "x"], rels, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], rels, 10) == 0.0


# ---- retrievers ----------------------------------------------------------
DOCS = ["the cat sat on the mat", "dogs chase cats in the park", "stock markets fell sharply today",
        "the cat and the dog are friends", "interest rates and stock prices"]


def naive_bm25(docs, query, k1=1.2, b=0.75):
    toks = [tokenize(d) for d in docs]
    n = len(docs)
    avg = sum(len(t) for t in toks) / n
    out = np.zeros(n)
    for term in tokenize(query):
        df = sum(1 for t in toks if term in t)
        if df == 0:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, t in enumerate(toks):
            tf = t.count(term)
            out[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avg))
    return out


def test_bm25_matches_naive_reference():
    bm = BM25().fit(DOCS)
    for q in ["cat dog", "stock rates", "cat cat mat", "unknownterm"]:
        assert np.allclose(bm.scores(q), naive_bm25(DOCS, q), atol=1e-5)


def test_top_k_ordering_and_ties():
    idx, sc = top_k(np.array([0.1, 0.9, 0.9, 0.5, 0.0]), 3)
    assert list(idx) == [1, 2, 3] and list(sc) == [0.9, 0.9, 0.5]


def test_top_k_drops_non_matching_documents():
    idx, _ = top_k(np.array([0.0, 0.4, 0.0, 0.2, 0.0]), 4, drop_zero=True)
    assert list(idx) == [1, 3]


def test_bm25_returns_only_matching_documents():
    idx, scores = BM25().fit(DOCS).search("stock", 5)
    assert set(idx) == {2, 4} and (scores > 0).all()


@pytest.mark.parametrize("model", [TfidfCosine(), BM25(), HashedRetriever(bits=12),
                                   HashedRetriever(bits=12, word_ngrams=(1, 1), char_ngrams=(3, 4))])
def test_retrievers_find_obvious_match(model):
    model.fit(DOCS)
    idx, _ = model.search("stock market prices", 2)
    assert set(idx) <= {2, 4}


def test_hashed_signed_and_unsigned_run():
    for signed in (True, False):
        m = HashedRetriever(bits=10, signed=signed).fit(DOCS)
        assert m.search("cat mat", 1)[0][0] == 0


# ---- end to end on the synthetic smoke-test dataset ---------------------
def test_end_to_end_on_toy_data(tmp_path):
    make(tmp_path)
    ds = load_beir(str(tmp_path), "toy")
    assert len(ds.query_ids) == 40
    result, run = run_retriever(BM25(), ds)
    # Only 6 of the 25 same-topic docs are judged relevant, so a perfect topic match
    # scores nDCG@10 of roughly 0.24, while a random ranking scores about 0.02.
    assert result["metrics"]["ndcg@10"] > 0.15
    assert result["latency_ms"]["mean"] > 0 and result["index_mb"] > 0
    assert len(run[ds.query_ids[0]]) == 100


def test_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(0)
    base = rng.random(200)
    res = paired_bootstrap(base + 0.2, base)
    assert res["p_bootstrap"] < 0.01 and res["ci95"][0] > 0
    assert paired_bootstrap(base, base)["p_bootstrap"] == 1.0
