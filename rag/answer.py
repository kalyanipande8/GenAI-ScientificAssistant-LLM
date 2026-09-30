"""Query-answering: retrieve real top-k documents, then compose a
structured answer via template-based synthesis of the retrieved chunks.

HONESTY NOTE: this is NOT a generative LLM writing prose. It is a
retrieval-augmented template composer -- the retrieval is real (computed
TF-IDF cosine similarity), but the "generation" step is rule-based string
assembly over the retrieved text, not a language model sampling tokens. See
README.md for why, and for exactly where a real Claude API call would slot
in if you wanted to upgrade the generation step later.
"""
from __future__ import annotations

from typing import List

from .index import RagStore
from .retrieve import RetrievedDoc, retrieve

# ---------------------------------------------------------------------------
# OPTIONAL / DISABLED live-LLM code path.
#
# This function is never called anywhere in this codebase. It exists only to
# show, concretely, the small change that would swap the local template
# synthesizer below for a real call to Anthropic's Claude API, using the
# official `anthropic` Python SDK. It is left disabled on purpose: this
# portfolio demo is scoped to be fully local/offline with no API key and no
# network calls or cost. Flip USE_LIVE_LLM to True and provide an API key to
# enable it -- nothing else in the pipeline needs to change, since
# synthesize_answer() below is the single call site generation flows through.
# ---------------------------------------------------------------------------
USE_LIVE_LLM = False  # kept False intentionally -- see note above


def _call_claude_api_DISABLED(query: str, context_chunks: List[str]) -> str:  # pragma: no cover
    """DISABLED by design. Not imported or invoked anywhere in this repo.

    Example of the small change that would enable a real hosted-LLM
    generation step:

        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        context = "\\n\\n".join(context_chunks)
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    f"Using ONLY the context below, answer the question.\\n\\n"
                    f"Context:\\n{context}\\n\\nQuestion: {query}"
                ),
            }],
        )
        return message.content[0].text

    That's the entire integration point -- synthesize_answer() would call
    this instead of _template_synthesize() when USE_LIVE_LLM is True.
    """
    raise RuntimeError(
        "Live LLM calls are disabled in this offline portfolio demo. "
        "This function is illustrative only and is never invoked."
    )


def _template_synthesize(query: str, hits: List[RetrievedDoc]) -> str:
    """Real, local, rule-based synthesis from retrieved chunks."""
    if not hits:
        return (
            f"No sufficiently relevant documents were found in the local "
            f"index for the query: '{query}'. Try rephrasing with more "
            f"specific assay, reagent, or organism terms."
        )

    lines = [f"Query: {query}", "", "Answer (synthesized from retrieved local documents):", ""]
    for i, hit in enumerate(hits, 1):
        doc = hit.document
        lines.append(
            f"[{i}] ({doc.doc_type}, relevance={hit.score:.3f}) {doc.title}\n"
            f"    {doc.text}"
        )
    lines.append("")
    top_types = sorted({h.document.doc_type for h in hits})
    lines.append(
        f"Summary: {len(hits)} relevant document(s) retrieved spanning "
        f"{', '.join(top_types)}. Highest-relevance match (score "
        f"{hits[0].score:.3f}): \"{hits[0].document.title}\". Review the "
        f"full excerpts above before acting on this information in a lab "
        f"setting."
    )
    return "\n".join(lines)


def synthesize_answer(store: RagStore, query: str, top_k: int = 5) -> str:
    hits = retrieve(store, query, top_k=top_k)
    if USE_LIVE_LLM:  # pragma: no cover -- intentionally dead in this demo
        return _call_claude_api_DISABLED(query, [h.document.text for h in hits])
    return _template_synthesize(query, hits)
