"""Download BEIR datasets: python scripts/download_beir.py scifact nfcorpus"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irstudy.data import download_beir, load_beir  # noqa: E402

if __name__ == "__main__":
    names = sys.argv[1:] or ["scifact", "nfcorpus"]
    for name in names:
        download_beir(name, "data")
        print(load_beir("data", name).summary())
