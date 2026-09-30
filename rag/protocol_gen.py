"""'Protocol generation' demo: given a target experiment type, retrieve
related protocol/SOP chunks from the local index and assemble a real
structured draft protocol document from them (template assembly, not a
generative LLM)."""
from __future__ import annotations

from datetime import date
from typing import List

from .index import RagStore
from .retrieve import RetrievedDoc, retrieve


def generate_protocol_draft(store: RagStore, experiment_type: str, top_k: int = 6) -> str:
    hits: List[RetrievedDoc] = retrieve(store, experiment_type, top_k=top_k)
    protocol_hits = [h for h in hits if h.document.doc_type in ("protocol_excerpt", "assay_sop")]
    other_hits = [h for h in hits if h not in protocol_hits]

    lines = [
        f"DRAFT PROTOCOL: {experiment_type}",
        f"Generated: {date.today().isoformat()} (local template assembly, no live LLM call)",
        "=" * 72,
        "",
        "1. PURPOSE",
        f"   Draft experimental protocol for '{experiment_type}', assembled from "
        f"the {len(protocol_hits)} most relevant protocol/SOP excerpts in the "
        f"local document index.",
        "",
        "2. MATERIALS & STEPS (retrieved from indexed protocol/SOP documents)",
    ]

    if not protocol_hits:
        lines.append("   No directly matching protocol/SOP documents were found; see related "
                      "results below and refine the query.")
    else:
        for i, hit in enumerate(protocol_hits, 1):
            doc = hit.document
            lines.append(f"   {i}. [{doc.doc_type}, relevance={hit.score:.3f}] {doc.title}")
            lines.append(f"      {doc.text}")

    lines.append("")
    lines.append("3. RELATED PRIOR RESULTS (context, retrieved from local index)")
    if other_hits:
        for hit in other_hits:
            lines.append(f"   - [{hit.score:.3f}] {hit.document.title}: {hit.document.text}")
    else:
        lines.append("   (none retrieved above threshold)")

    lines.append("")
    lines.append("4. REVIEW NOTE")
    lines.append(
        "   This draft was assembled automatically from retrieved local text "
        "chunks via string templates -- a human (PI/lab lead) must review, "
        "reconcile conflicting steps across sources, and approve before use "
        "at the bench."
    )
    return "\n".join(lines)
