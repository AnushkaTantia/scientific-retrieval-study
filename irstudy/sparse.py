"""Lexical retrievers: BM25 (own vectorised implementation) and TF-IDF cosine.

All retrievers share one interface:
    fit(doc_texts)            build the index
    search(query, k)          -> (doc indices, scores), best first
    index_bytes()             size of the in-memory index
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .text import tokenize


def top_k(scores: np.ndarray, k: int, drop_zero: bool = False):
    """Indices and scores of the k highest scores, ties broken by document order.

    With ``drop_zero`` documents scoring exactly 0 (no feature shared with the query) are
    not returned, as in Lucene. Without it, a query matching fewer than k documents would be
    padded with arbitrary zero-score documents, and which ones depends on the numpy version.
    """
    order = np.argsort(-scores, kind="stable")
    if drop_zero:
        order = order[scores[order] != 0]
    idx = order[:k]
    return idx, scores[idx]


def csr_bytes(m: csr_matrix) -> int:
    return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes


class BM25:
    """Okapi BM25 with the Lucene idf, log(1 + (N - n + 0.5) / (n + 0.5)).

    The document-side weight of every (document, term) pair is precomputed into a
    sparse matrix, so a query is a single sparse product.
    """

    def __init__(self, k1=1.2, b=0.75, stem=False, stop=True):
        self.k1, self.b, self.stem, self.stop = k1, b, stem, stop
        self.name = "bm25" + ("-stem" if stem else "")

    def _analyzer(self, text):
        return tokenize(text, stop=self.stop, stem=self.stem)

    def fit(self, doc_texts):
        self.vec = CountVectorizer(analyzer=self._analyzer)
        tf = self.vec.fit_transform(doc_texts).tocsr().astype(np.float64)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = doc_len.mean()
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        norm = self.k1 * (1.0 - self.b + self.b * doc_len[rows] / avg_len)
        weights = tf.data * (self.k1 + 1.0) / (tf.data + norm) * self.idf[tf.indices]
        w = csr_matrix((weights, tf.indices, tf.indptr), shape=tf.shape)
        self.term_doc = w.T.tocsr().astype(np.float32)  # terms x docs
        return self

    def scores(self, query):
        q = self.vec.transform([query])  # query term counts, unknown terms dropped
        return np.asarray((q @ self.term_doc).todense()).ravel()

    def search(self, query, k=100):
        return top_k(self.scores(query), k, drop_zero=True)

    def index_bytes(self):
        return csr_bytes(self.term_doc)


class TfidfCosine:
    """TF-IDF (sublinear tf, smoothed idf, L2-normalised) with cosine similarity."""

    def __init__(self, ngram_range=(1, 1), stem=False, stop=True):
        self.ngram_range, self.stem, self.stop = ngram_range, stem, stop
        self.name = "tfidf"

    def _analyzer(self, text):
        toks = tokenize(text, stop=self.stop, stem=self.stem)
        lo, hi = self.ngram_range
        out = []
        for n in range(lo, hi + 1):
            out.extend(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
        return out

    def fit(self, doc_texts):
        self.vec = TfidfVectorizer(analyzer=self._analyzer, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(doc_texts).tocsr().astype(np.float32)
        self.matrix_t = self.matrix.T.tocsr()
        return self

    def scores(self, query):
        q = self.vec.transform([query])
        return np.asarray((q @ self.matrix_t).todense()).ravel()

    def search(self, query, k=100):
        return top_k(self.scores(query), k, drop_zero=True)

    def index_bytes(self):
        return csr_bytes(self.matrix_t)
