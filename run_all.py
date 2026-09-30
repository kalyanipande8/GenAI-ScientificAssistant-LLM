#!/usr/bin/env python3
"""End-to-end demo: build the local index over 500+ synthetic scientific
documents, then run several real example queries and a protocol-generation
example, printing real retrieved results (not hardcoded).

Run: python3 run_all.py
"""
from __future__ import annotations

from rag.answer import synthesize_answer
from rag.index import build_and_save
from rag.protocol_gen import generate_protocol_draft

EXAMPLE_QUERIES = [
    "How do I quantify mitochondrial DNA copy number by qPCR in zebrafish embryos?",
    "What is the standard procedure for running an ELISA assay?",
    "Show me results about apoptosis induction measured by flow cytometry",
    "protocol for CRISPR knockout screening in HEK293T cells",
]

EXAMPLE_PROTOCOL_TARGETS = [
    "Western blot protocol for detecting protein expression in Jurkat T cells",
]


def main() -> None:
    print("=" * 72)
    print("Building local TF-IDF index over synthetic scientific document corpus")
    print("=" * 72)
    store = build_and_save(n_docs=520, seed=42)
    print(f"Indexed {len(store.documents)} documents "
          f"({len(store.index.doc_freq)} unique vocabulary terms).")
    print("No network calls made. No API key used. 100% local computation.\n")

    for q in EXAMPLE_QUERIES:
        print("-" * 72)
        print(synthesize_answer(store, q, top_k=3))
        print()

    print("=" * 72)
    print("PROTOCOL GENERATION DEMO")
    print("=" * 72)
    for target in EXAMPLE_PROTOCOL_TARGETS:
        print(generate_protocol_draft(store, target, top_k=4))
        print()


if __name__ == "__main__":
    main()
