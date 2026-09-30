"""Build, save, and load the local vector index over the synthetic
scientific document corpus."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List

from .corpus import Document, build_corpus
from .embeddings import TfidfIndex

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.json")
DOCS_PATH = os.path.join(DATA_DIR, "documents.json")


class RagStore:
    """Holds the document store + TF-IDF index together."""

    def __init__(self, documents: List[Document], index: TfidfIndex):
        self.documents: Dict[str, Document] = {d.doc_id: d for d in documents}
        self.index = index

    def get(self, doc_id: str) -> Document:
        return self.documents[doc_id]


def build_and_save(n_docs: int = 520, seed: int = 42) -> RagStore:
    os.makedirs(DATA_DIR, exist_ok=True)
    docs = build_corpus(n=n_docs, seed=seed)
    index = TfidfIndex()
    index.fit([(d.doc_id, d.title + ". " + d.text) for d in docs])

    with open(DOCS_PATH, "w") as f:
        json.dump([asdict(d) for d in docs], f, indent=2)
    with open(INDEX_PATH, "w") as f:
        json.dump(index.to_dict(), f)

    return RagStore(docs, index)


def load() -> RagStore:
    if not (os.path.exists(DOCS_PATH) and os.path.exists(INDEX_PATH)):
        return build_and_save()
    with open(DOCS_PATH) as f:
        raw_docs = json.load(f)
    docs = [Document(**d) for d in raw_docs]
    with open(INDEX_PATH) as f:
        index = TfidfIndex.from_dict(json.load(f))
    return RagStore(docs, index)
