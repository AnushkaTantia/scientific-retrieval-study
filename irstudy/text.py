"""Shared text normalisation used by every retriever, so comparisons are fair."""
import re
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"[a-z0-9]+")

try:
    from nltk.stem import PorterStemmer

    _stemmer = PorterStemmer()
except ImportError:  # stemming is optional
    _stemmer = None


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def tokenize(text: str, stop: bool = True, stem: bool = False) -> list:
    """Lower-case, split on non-alphanumerics, optionally drop stop words and stem."""
    tokens = _TOKEN.findall(text.lower())
    if stop:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    if stem:
        if _stemmer is None:
            raise RuntimeError("stemming requested but nltk is not installed")
        tokens = [_stem(t) for t in tokens]
    return tokens


def normalise_chars(text: str) -> str:
    """Lower-case and collapse everything that is not a letter or digit to one space."""
    return " ".join(_TOKEN.findall(text.lower()))
