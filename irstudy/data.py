"""Loading BEIR-format datasets (corpus.jsonl, queries.jsonl, qrels/<split>.tsv)."""
import csv
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


@dataclass
class Dataset:
    name: str
    doc_ids: list
    doc_texts: list  # title + " " + text
    query_ids: list
    query_texts: list
    qrels: dict  # query id -> {doc id: graded relevance (> 0 only)}

    def summary(self) -> str:
        n_rel = sum(len(v) for v in self.qrels.values())
        return (f"{self.name}: {len(self.doc_ids)} docs, {len(self.query_ids)} queries, "
                f"{n_rel} relevance judgements")


def download_beir(name: str, data_dir: str = "data") -> Path:
    """Download and unzip one BEIR dataset (needs access to the BEIR host)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / name
    if (target / "corpus.jsonl").exists():
        return target
    zip_path = data_dir / f"{name}.zip"
    urllib.request.urlretrieve(BEIR_URL.format(name), zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    zip_path.unlink()
    return target


def load_beir(data_dir: str, name: str, split: str = "test") -> Dataset:
    root = Path(data_dir) / name
    doc_ids, doc_texts = [], []
    with open(root / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_ids.append(str(row["_id"]))
            doc_texts.append((row.get("title", "") + " " + row.get("text", "")).strip())

    all_queries = {}
    with open(root / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            all_queries[str(row["_id"])] = row["text"]

    qrels = {}
    with open(root / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(str(qid), {})[str(did)] = int(score)

    # BEIR evaluates only queries that have at least one relevant document
    query_ids = [q for q in all_queries if q in qrels]
    return Dataset(name, doc_ids, doc_texts, query_ids,
                   [all_queries[q] for q in query_ids],
                   {q: qrels[q] for q in query_ids})
