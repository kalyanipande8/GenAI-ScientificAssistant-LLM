"""A real, from-scratch TF-IDF embedding + cosine-similarity retrieval
engine, implemented with the Python standard library only (no numpy, no
scikit-learn, no external services).

This is a legitimate, if small-scale, vector-space model: documents are
embedded as sparse TF-IDF vectors over the corpus vocabulary, and retrieval
ranks documents by cosine similarity to the query's TF-IDF vector -- the
same mathematical approach scikit-learn's TfidfVectorizer + a nearest
neighbor search would give you, just written by hand because this
environment could not install extra packages (see README for details).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]+")

_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "to", "of",
    "in", "on", "for", "and", "or", "with", "at", "by", "from", "this",
    "that", "these", "those", "as", "it", "its", "into", "using", "used",
    "each", "all", "any", "not", "no", "than", "then", "over", "under",
    "such", "but", "which", "will", "can", "may", "must", "should",
}


def tokenize(text: str) -> List[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class TfidfIndex:
    """A hand-rolled TF-IDF vector index with cosine-similarity search.

    Vectors are stored as sparse dicts {term: weight}. This avoids needing
    numpy while still doing real linear-algebra-equivalent similarity
    scoring (dot product over sparse dicts, normalized by L2 norm).
    """

    def __init__(self) -> None:
        self.doc_ids: List[str] = []
        self.doc_freq: Counter = Counter()
        self.n_docs: int = 0
        self.vectors: Dict[str, Dict[str, float]] = {}
        self.norms: Dict[str, float] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[Tuple[str, str]]) -> None:
        """documents: list of (doc_id, text)."""
        self.n_docs = len(documents)
        term_docs: Dict[str, set] = defaultdict(set)
        raw_tf: Dict[str, Counter] = {}

        for doc_id, text in documents:
            tokens = tokenize(text)
            counts = Counter(tokens)
            raw_tf[doc_id] = counts
            for term in counts:
                term_docs[term].add(doc_id)
            self.doc_ids.append(doc_id)

        self.doc_freq = Counter({term: len(ids) for term, ids in term_docs.items()})
        # Smoothed IDF, same formula sklearn's TfidfVectorizer uses by default.
        self.idf = {
            term: math.log((1 + self.n_docs) / (1 + df)) + 1.0
            for term, df in self.doc_freq.items()
        }

        for doc_id, counts in raw_tf.items():
            total = sum(counts.values()) or 1
            vec: Dict[str, float] = {}
            for term, count in counts.items():
                tf = count / total
                vec[term] = tf * self.idf[term]
            norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
            self.vectors[doc_id] = vec
            self.norms[doc_id] = norm

    def embed_query(self, text: str) -> Dict[str, float]:
        tokens = tokenize(text)
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        vec: Dict[str, float] = {}
        for term, count in counts.items():
            if term in self.idf:
                tf = count / total
                vec[term] = tf * self.idf[term]
        return vec

    @staticmethod
    def _cosine(vec_a: Dict[str, float], norm_a: float,
                vec_b: Dict[str, float], norm_b: float) -> float:
        if norm_a == 0 or norm_b == 0:
            return 0.0
        # dot product over the smaller vector's keys
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        dot = sum(w * vec_b.get(term, 0.0) for term, w in vec_a.items())
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        q_vec = self.embed_query(query)
        q_norm = math.sqrt(sum(w * w for w in q_vec.values())) or 1.0
        scores = []
        for doc_id in self.doc_ids:
            score = self._cosine(q_vec, q_norm, self.vectors[doc_id], self.norms[doc_id])
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def to_dict(self) -> dict:
        return {
            "doc_ids": self.doc_ids,
            "n_docs": self.n_docs,
            "idf": self.idf,
            "vectors": self.vectors,
            "norms": self.norms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TfidfIndex":
        obj = cls()
        obj.doc_ids = data["doc_ids"]
        obj.n_docs = data["n_docs"]
        obj.idf = data["idf"]
        obj.vectors = data["vectors"]
        obj.norms = data["norms"]
        return obj
