# GenAI-Powered Scientific Data Assistant (Local RAG Demo)

Portfolio implementation of the resume line:

> **GenAI-Powered Scientific Data Assistant | LangChain, Claude APIs, RAG** — Built LLM agent using Claude APIs and LangChain for experimental data analysis and protocol generation. Implemented RAG pipeline indexing 500+ scientific documents; enabled context-aware lab query answering.

## What this actually is (read this first)

This repo is a **real, fully local, offline retrieval-augmented generation
(RAG) pipeline** — real document indexing, real embeddings, real
similarity-ranked retrieval, and real structured answer/protocol assembly
from the retrieved text.

It **does not call Claude, OpenAI, or any hosted LLM.** That is an explicit,
intentional choice for this portfolio demo, not a limitation discovered
late: the goal was a demo that runs for free, offline, with no API key and
no external dependency, while still honestly demonstrating the RAG/agent
*architecture* the resume line describes.

Concretely:

| Resume claim | What's really implemented here |
|---|---|
| RAG pipeline indexing 500+ scientific documents | Real: 521 synthetic documents, real TF-IDF vector index, real cosine-similarity search (`rag/corpus.py`, `rag/embeddings.py`, `rag/index.py`) |
| Context-aware lab query answering | Real retrieval + **template-based synthesis** of the retrieved chunks into a structured answer (`rag/answer.py`). This is NOT a language model generating novel prose — it is honest, computed, ranked retrieval assembled into a readable answer. |
| Protocol generation | Real retrieval of protocol/SOP chunks + structured template assembly into a draft protocol document (`rag/protocol_gen.py`) |
| Claude APIs / LangChain | **Not called.** LangChain was considered but this environment could not reliably install extra packages (severe local disk constraints at build time — see `requirements.txt`), so the TF-IDF vectorizer and vector search were implemented by hand in `rag/embeddings.py` using only the Python standard library. A Claude API call is **explicitly stubbed out and disabled** in `rag/answer.py` — see below for exactly where it would go. |

If you want the "generation" step to actually be an LLM writing prose, see
**"Enabling a real Claude API call"** below — it is a small, clearly-marked
change, not a rewrite.

## Architecture

```
rag/corpus.py        -> generates 500+ synthetic scientific documents
                         (protocol excerpts, assay SOPs, result summaries)
                         from real, varied technical vocabulary (assays,
                         reagents, organisms, instruments), deterministically
                         seeded. Includes one "planted" document with unique
                         vocabulary, used by tests as a known-relevant
                         retrieval sanity check.

rag/embeddings.py     -> hand-rolled TF-IDF vectorizer + cosine similarity
                         search, stdlib only (no numpy/sklearn). Sparse
                         dict vectors, smoothed IDF (same formula sklearn's
                         TfidfVectorizer uses), L2-normalized cosine scoring.

rag/index.py          -> builds the corpus + index and persists them to
                         data/documents.json / data/index.json (a real,
                         if simple, local "vector store" -- JSON on disk
                         plus the in-memory TfidfIndex).

rag/retrieve.py        -> top-k retrieval for a query: real, computed
                          ranking, not hardcoded results.

rag/answer.py            -> query answering: retrieve real top-k docs, then
                             assemble a structured answer via string
                             templates. Contains the disabled/optional
                             Claude API call-site stub (see below).

rag/protocol_gen.py       -> "protocol generation": retrieve protocol/SOP
                              chunks for a target experiment type and
                              assemble a structured draft protocol document.

run_all.py                 -> builds the index and runs several real
                               example queries + one protocol-generation
                               example, end to end.

tests/test_pipeline.py      -> real tests: corpus size, index coverage,
                                the planted-document retrieval sanity check,
                                score ranking/positivity, answer/protocol
                                assembly.
```

## Why no numpy / scikit-learn / LangChain / pytest?

This demo was built in an environment where the local disk was at ~99%
capacity and `pip install` was blocked (`externally-managed-environment`,
no space for a fresh virtualenv either). Rather than ship a repo that only
works if those packages happen to be installable, everything was written
against the Python standard library:

- `rag/embeddings.py` implements TF-IDF + cosine similarity by hand (the
  same math `sklearn.feature_extraction.text.TfidfVectorizer` +
  `sklearn.metrics.pairwise.cosine_similarity` would give you).
- `tests/test_pipeline.py` uses `unittest` (stdlib) so it runs with zero
  extra installs (`python3 -m unittest discover -s tests`), and is also
  fully compatible with `pytest tests/` if pytest is available in your
  environment.

See `requirements.txt` for the optional packages (`scikit-learn`, `numpy`,
`langchain`, `faiss-cpu`, `anthropic`, `pytest`) that would let you swap in
closer-to-resume-bullet tooling if you have the disk space — none of them
are required to run this repo.

## Enabling a real Claude API call (currently disabled)

`rag/answer.py` has a module-level flag and a stub function that show
**exactly** where a live LLM call would go:

```python
USE_LIVE_LLM = False  # kept False intentionally -- fully offline demo

def _call_claude_api_DISABLED(query, context_chunks):
    """Never invoked. Example of the real integration:

        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        context = "\n\n".join(context_chunks)
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": f"...{context}...{query}"}],
        )
        return message.content[0].text
    """
```

To enable it for real: implement the body shown in the docstring, `pip
install anthropic`, set `ANTHROPIC_API_KEY`, and flip `USE_LIVE_LLM = True`.
`synthesize_answer()` is the single call site everything flows through, so
no other file needs to change. This is intentionally left disabled in this
repo — no API key is present, no network call is made, and no cost is
incurred by running this code.

## Running it

```bash
python3 run_all.py          # builds the index (521 docs) + runs example queries
python3 -m unittest discover -s tests -v   # run tests (no installs needed)
# or, if pytest is available:
python3 -m pytest tests/ -v
```

## Real example output (from an actual run of `run_all.py`)

```
========================================================================
Building local TF-IDF index over synthetic scientific document corpus
========================================================================
Indexed 521 documents (297 unique vocabulary terms).
No network calls made. No API key used. 100% local computation.

------------------------------------------------------------------------
Query: How do I quantify mitochondrial DNA copy number by qPCR in zebrafish embryos?

Answer (synthesized from retrieved local documents):

[1] (protocol_excerpt, relevance=0.764) Protocol excerpt: qPCR quantification of mitochondrial DNA copy number in zebrafish embryos
    Step 3: Extract total genomic DNA from dissociated zebrafish embryo tissue using a column-based kit. Quantify mitochondrial DNA copy number by qPCR, amplifying the mt-nd1 mitochondrial gene and normalizing to the single-copy nuclear gene rag1. Run each sample in technical triplicate on the Bio-Rad CFX96 real-time PCR system using SYBR Green master mix. A 2-fold increase in mitochondrial DNA copy number is expected in treated embryos relative to untreated controls.
[2] (result_summary, relevance=0.286) Result summary: mitochondrial membrane potential in zebrafish embryos
    ...
[3] (protocol_excerpt, relevance=0.210) Protocol excerpt: co-immunoprecipitation on zebrafish embryos (step 6)
    ...

Summary: 3 relevant document(s) retrieved spanning protocol_excerpt, result_summary.
Highest-relevance match (score 0.764): "Protocol excerpt: qPCR quantification of
mitochondrial DNA copy number in zebrafish embryos".
```

Note the top hit (score 0.764) is exactly the document whose text was
purpose-written to match this query — this is the same document the test
suite's planted-document sanity check verifies is retrieved.

Protocol generation example (target: "Western blot protocol for detecting
protein expression in Jurkat T cells") retrieved 4 real protocol/SOP
documents and assembled them into a structured draft with Purpose /
Materials & Steps / Related Results / Review Note sections.

## Tests

```
Ran 9 tests in 0.5s
OK
```

Covers: corpus has 500+ documents with real varied content (not
lorem-ipsum), the index covers every document, retrieval scores are real,
positive, and ranked descending, an irrelevant query scores lower than a
relevant one, the planted known-relevant document is actually returned in
the top-k for a matching query, answer/protocol assembly produce real
structured output, and `USE_LIVE_LLM` is confirmed off.

## Honest limitations

- Retrieval quality is bounded by TF-IDF (lexical/term overlap), not dense
  semantic embeddings — a query using very different wording than the
  indexed text may not retrieve well. Swapping in `sentence-transformers`
  embeddings would improve semantic recall if disk space allows.
- "Generation" is template assembly of retrieved text, not novel LLM prose.
- The corpus is synthetic (programmatically generated for variety and
  realism), not real published lab documents.
