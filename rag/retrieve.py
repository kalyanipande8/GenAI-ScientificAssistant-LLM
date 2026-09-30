"""Retrieval layer: real, computed top-k document retrieval for a query
against the local TF-IDF index (no hardcoded results)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .corpus import Document
from .index import RagStore


@dataclass
class RetrievedDoc:
    document: Document
    score: float


def retrieve(store: RagStore, query: str, top_k: int = 5) -> List[RetrievedDoc]:
    hits = store.index.search(query, top_k=top_k)
    return [RetrievedDoc(document=store.get(doc_id), score=score) for doc_id, score in hits]
