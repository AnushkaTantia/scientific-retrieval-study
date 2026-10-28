"""Feature-hashing retriever (own implementation).

Every word n-gram and/or character n-gram is mapped to one of 2**bits buckets with
MurmurHash3, optionally with a random sign (so collisions cancel in expectation).
Bucket counts get sublinear tf, bucket-level idf and L2 normalisation, and documents are
ranked by cosine similarity. There is no vocabulary to store, so the index size is bounded
by the number of buckets, which is what the accuracy / memory trade-off study varies.
"""
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils import murmurhash3_32

from .sparse import csr_bytes, top_k
from .text import normalise_chars, tokenize

_HASH_CACHE = {}


def hash32(feature: str, seed: int = 0) -> int:
    key = (feature, seed)
    h = _HASH_CACHE.get(key)
    if h is None:
        h = murmurhash3_32(feature, seed=seed, positive=True)
        _HASH_CACHE[key] = h
    return h


class FeatureExtractor:
    """Turns text into a Counter of string features (word and/or character n-grams)."""

    def __init__(self, word_ngrams=(1, 2), char_ngrams=None, stem=False, stop=True):
        self.word_ngrams, self.char_ngrams = word_ngrams, char_ngrams
        self.stem, self.stop = stem, stop

    def key(self):
        return (self.word_ngrams, self.char_ngrams, self.stem, self.stop)

    def __call__(self, text):
        feats = Counter()
        if self.word_ngrams:
            toks = tokenize(text, stop=self.stop, stem=self.stem)
            lo, hi = self.word_ngrams
            for n in range(lo, hi + 1):
                for i in range(len(toks) - n + 1):
                    feats["w:" + " ".join(toks[i:i + n])] += 1
        if self.char_ngrams:
            lo, hi = self.char_ngrams
            for word in normalise_chars(text).split():
                padded = f" {word} "
                for n in range(lo, hi + 1):
                    for i in range(max(len(padded) - n + 1, 1)):
                        feats["c:" + padded[i:i + n]] += 1
        return feats


class HashedRetriever:
    def __init__(self, bits=14, word_ngrams=(1, 2), char_ngrams=None, signed=True,
                 stem=False, stop=True, seed=0):
        self.bits, self.signed, self.seed = bits, signed, seed
        self.extractor = FeatureExtractor(word_ngrams, char_ngrams, stem, stop)
        parts = []
        if word_ngrams:
            parts.append(f"w{word_ngrams[0]}-{word_ngrams[1]}")
        if char_ngrams:
            parts.append(f"c{char_ngrams[0]}-{char_ngrams[1]}")
        self.name = f"hashed-{'+'.join(parts)}-b{bits}"

    # -- vectorising -------------------------------------------------------
    def _bucket(self, feature):
        h = hash32(feature, self.seed)
        idx = h & ((1 << self.bits) - 1)
        sign = (1 - 2 * ((h >> 31) & 1)) if self.signed else 1
        return idx, sign

    def _to_row(self, counter):
        acc = {}
        for feat, c in counter.items():
            idx, sign = self._bucket(feat)
            acc[idx] = acc.get(idx, 0) + sign * c
        idx = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
        val = np.fromiter(acc.values(), dtype=np.float64, count=len(acc))
        keep = val != 0
        return idx[keep], val[keep]

    def _matrix(self, counters, idf=None):
        indptr, indices, data = [0], [], []
        for counter in counters:
            idx, val = self._to_row(counter)
            val = np.sign(val) * np.log1p(np.abs(val))  # sublinear tf
            if idf is not None:
                val = val * idf[idx]
            norm = np.linalg.norm(val)
            if norm > 0:
                val = val / norm
            indices.append(idx)
            data.append(val)
            indptr.append(indptr[-1] + len(idx))
        return csr_matrix(
            (np.concatenate(data) if data else np.zeros(0),
             np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64), indptr),
            shape=(len(indptr) - 1, 1 << self.bits))

    # -- retriever interface ----------------------------------------------
    def extract(self, doc_texts):
        """Feature extraction is the slow part; callers can reuse it across bucket sizes."""
        return [self.extractor(t) for t in doc_texts]

    def fit(self, doc_texts=None, counters=None):
        counters = counters if counters is not None else self.extract(doc_texts)
        raw = self._matrix(counters)
        n_docs = raw.shape[0]
        df = np.bincount(raw.indices, minlength=1 << self.bits)
        self.idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        self.matrix_t = self._matrix(counters, idf=self.idf).astype(np.float32).T.tocsr()
        return self

    def scores(self, query):
        q = self._matrix([self.extractor(query)], idf=self.idf).astype(np.float32)
        return np.asarray((q @ self.matrix_t).todense()).ravel()

    def search(self, query, k=100):
        return top_k(self.scores(query), k, drop_zero=True)

    def index_bytes(self):
        return csr_bytes(self.matrix_t)
