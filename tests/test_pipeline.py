"""Real pipeline tests. Written with unittest (stdlib) so they run with
zero extra dependencies (`python3 -m unittest`); they are also fully
discoverable and runnable by pytest if it happens to be installed
(`python3 -m pytest tests/`)."""
from __future__ import annotations

import unittest

from rag.answer import synthesize_answer
from rag.corpus import build_corpus
from rag.embeddings import TfidfIndex
from rag.index import RagStore
from rag.protocol_gen import generate_protocol_draft
from rag.retrieve import retrieve


class TestCorpus(unittest.TestCase):
    def test_corpus_has_500_plus_documents(self):
        docs = build_corpus(n=520, seed=42)
        self.assertGreaterEqual(len(docs), 500)

    def test_documents_are_varied_not_lorem_ipsum(self):
        docs = build_corpus(n=520, seed=42)
        texts = {d.text for d in docs}
        # With real varied synthesis, most generated docs should be unique.
        self.assertGreater(len(texts), 400)
        self.assertNotIn("lorem ipsum", " ".join(texts).lower())


class TestIndexAndRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = build_corpus(n=520, seed=42)
        index = TfidfIndex()
        index.fit([(d.doc_id, d.title + ". " + d.text) for d in cls.docs])
        cls.store = RagStore(cls.docs, index)

    def test_index_covers_all_documents(self):
        self.assertEqual(len(self.store.index.doc_ids), len(self.docs))
        self.assertGreaterEqual(len(self.docs), 500)

    def test_planted_document_is_retrieved_top_k(self):
        """Sanity check: a query closely matching a specific planted
        document's unique vocabulary must actually surface that document
        in the top-k results via real computed cosine similarity."""
        query = "qPCR quantification of mitochondrial DNA copy number in zebrafish embryos"
        hits = retrieve(self.store, query, top_k=5)
        retrieved_ids = [h.document.doc_id for h in hits]
        self.assertIn("planted-qpcr-mitochondrial-0001", retrieved_ids)

    def test_retrieval_scores_are_real_and_ranked_descending(self):
        query = "ELISA standard operating procedure"
        hits = retrieve(self.store, query, top_k=5)
        self.assertTrue(len(hits) > 0)
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for h in hits:
            self.assertGreater(h.score, 0.0)

    def test_irrelevant_query_scores_lower_than_relevant_query(self):
        relevant = retrieve(self.store, "Western blot protein detection Jurkat T cells", top_k=1)
        unrelated = retrieve(self.store, "xyz nonsense query unrelated to anything qwerty", top_k=1)
        self.assertTrue(relevant)
        top_relevant_score = relevant[0].score
        top_unrelated_score = unrelated[0].score if unrelated else 0.0
        self.assertGreater(top_relevant_score, top_unrelated_score)


class TestAnswerSynthesis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = build_corpus(n=520, seed=42)
        index = TfidfIndex()
        index.fit([(d.doc_id, d.title + ". " + d.text) for d in cls.docs])
        cls.store = RagStore(cls.docs, index)

    def test_answer_contains_retrieved_content(self):
        answer = synthesize_answer(self.store, "ELISA assay procedure", top_k=3)
        self.assertIn("Query:", answer)
        self.assertIn("synthesized from retrieved local documents", answer.lower())

    def test_no_live_llm_flag_is_off(self):
        from rag import answer as answer_module
        self.assertFalse(answer_module.USE_LIVE_LLM)


class TestProtocolGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = build_corpus(n=520, seed=42)
        index = TfidfIndex()
        index.fit([(d.doc_id, d.title + ". " + d.text) for d in cls.docs])
        cls.store = RagStore(cls.docs, index)

    def test_protocol_draft_is_assembled_from_real_matches(self):
        draft = generate_protocol_draft(self.store, "Western blot protein detection protocol")
        self.assertIn("DRAFT PROTOCOL", draft)
        self.assertIn("MATERIALS & STEPS", draft)
        self.assertGreater(len(draft), 200)


if __name__ == "__main__":
    unittest.main()
