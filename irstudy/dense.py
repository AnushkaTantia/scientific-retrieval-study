"""Dense bi-encoder retrieval with an exact FAISS inner-product index.

Embeddings are L2-normalised, so inner product equals cosine similarity. Document
embeddings are cached to disk so the expensive encoding step runs only once.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

# torch and faiss each ship an OpenMP runtime on Windows; without this they can abort
# with "OMP: Error #15" when both are loaded in one process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss  # noqa: E402

MODELS = {
    # short name: (HuggingFace id, prefix added to queries only)
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5",
                  "Represent this sentence for searching relevant passages: "),
}


def load_encoder(hf_id):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(hf_id, device="cpu")


class DenseRetriever:
    def __init__(self, short_name, cache_dir, encoder=None, batch_size=32):
        self.short = short_name
        self.hf_id, self.query_prefix = MODELS[short_name]
        self.name = f"dense-{short_name}"
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self._encoder = encoder
        self.encode_seconds = None  # time to embed the corpus (None if loaded from cache)

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = load_encoder(self.hf_id)
        return self._encoder

    def encode(self, texts, prefix="", show_progress=False):
        emb = self.encoder.encode([prefix + t for t in texts], batch_size=self.batch_size,
                                  normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=show_progress)
        return np.ascontiguousarray(emb, dtype=np.float32)

    def fit(self, doc_texts, doc_ids=None):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path = self.cache_dir / f"{self.short}__docs.npy"
        ids_path = self.cache_dir / f"{self.short}__doc_ids.json"
        if emb_path.exists():
            self.doc_emb = np.load(emb_path)
        else:
            t0 = time.perf_counter()
            self.doc_emb = self.encode(doc_texts, show_progress=True)
            self.encode_seconds = time.perf_counter() - t0
            np.save(emb_path, self.doc_emb)
            if doc_ids is not None:
                ids_path.write_text(json.dumps(doc_ids))
            (self.cache_dir / f"{self.short}__encode_seconds.json").write_text(
                json.dumps({"corpus_encode_seconds": self.encode_seconds,
                            "n_docs": len(doc_texts)}))
        if len(self.doc_emb) != len(doc_texts):
            raise ValueError(f"cached embeddings in {emb_path} do not match the corpus; delete them")
        self.index = faiss.IndexFlatIP(self.doc_emb.shape[1])
        self.index.add(self.doc_emb)
        return self

    def search(self, query, k=100):
        """Includes query encoding time, as a deployed system would pay it."""
        q = self.encode([query], prefix=self.query_prefix)
        scores, idx = self.index.search(q, k)
        return idx[0], scores[0]

    def encode_queries(self, query_texts):
        return self.encode(query_texts, prefix=self.query_prefix)

    def index_bytes(self):
        return self.doc_emb.nbytes
