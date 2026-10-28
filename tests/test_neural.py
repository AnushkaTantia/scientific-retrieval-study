import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irstudy.fusion import minmax, ranks_from_scores, rrf, weighted_fusion  # noqa: E402
from irstudy.rerank import rerank  # noqa: E402


def test_minmax_and_constant_vector():
    assert np.allclose(minmax(np.array([2.0, 4.0, 3.0])), [0, 1, 0.5])
    assert np.allclose(minmax(np.array([5.0, 5.0])), [0, 0])


def test_weighted_fusion_endpoints():
    lex, den = np.array([3.0, 1.0, 2.0]), np.array([0.1, 0.9, 0.5])
    assert np.argmax(weighted_fusion(lex, den, 0.0)) == 0  # pure lexical
    assert np.argmax(weighted_fusion(lex, den, 1.0)) == 1  # pure dense
    assert np.allclose(weighted_fusion(lex, den, 0.5), 0.5 * minmax(lex) + 0.5 * minmax(den))


def test_fusion_normalises_each_query_separately():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_analysis import fuse_rows
    L = np.array([[1.0, 2.0, 3.0], [100.0, 300.0, 200.0]])
    D = np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.3]])
    H = fuse_rows(L, D, 0.5)
    for i in range(2):
        assert np.allclose(H[i], weighted_fusion(L[i], D[i], 0.5))
    # every query's fused scores span [0, 1] on their own
    # query 2: lexical [100, 300, 200] -> [0, 1, 0.5]; dense [.1, .2, .3] -> [0, .5, 1]
    assert np.allclose(H[1], [0.0, 0.75, 0.75])
    assert np.allclose(H[0], [0.5, 0.5, 0.5])  # query 1: the two signals cancel exactly


def test_rrf_by_hand():
    a = np.array([0.9, 0.5, 0.1])  # ranks 1, 2, 3
    b = np.array([0.2, 0.8, 0.5])  # ranks 3, 1, 2
    expected = [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62]
    assert np.allclose(rrf([a, b], k=60), expected)


def test_rrf_depth_cutoff():
    r = ranks_from_scores(np.array([0.3, 0.9, 0.1, 0.5]), depth=2)
    assert r[1] == 1 and r[3] == 2 and np.isinf(r[0]) and np.isinf(r[2])


def test_rerank_only_touches_head():
    ranking = ["a", "b", "c", "d", "e"]
    ce = {"a": 0.1, "b": 0.9, "c": 0.5}
    assert rerank(ranking, ce, 3) == ["b", "c", "a", "d", "e"]


@pytest.mark.filterwarnings("ignore")
def test_dense_retriever_with_stub_encoder(tmp_path):
    pytest.importorskip("faiss")
    from irstudy.dense import DenseRetriever

    class Stub:
        def encode(self, texts, **kw):
            m = np.array([[t.count("cat"), t.count("stock"), 1.0] for t in texts], dtype=np.float32)
            return m / np.linalg.norm(m, axis=1, keepdims=True)

    docs = ["cat cat cat", "stock stock stock", "nothing here"]
    d = DenseRetriever("minilm", tmp_path, encoder=Stub()).fit(docs, ["0", "1", "2"])
    idx, scores = d.search("stock", 2)
    assert idx[0] == 1 and scores[0] >= scores[1]
    assert (tmp_path / "minilm__docs.npy").exists()
    # second fit loads the cached embeddings instead of re-encoding
    d2 = DenseRetriever("minilm", tmp_path, encoder=None).fit(docs)
    assert np.allclose(d2.doc_emb, d.doc_emb)
