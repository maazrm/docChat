import pytest
from unittest.mock import patch, MagicMock

from pipeline.agents import retriever as retriever_module
from pipeline.agents.retriever import run, _retrieve_for_query
from pipeline.state import PipelineState


def _make_state(**overrides) -> PipelineState:
    state: PipelineState = {
        "doc_id": "doc123",
        "topic_summary": "Test document about AI.",
        "raw_query": "What is machine learning?",
        "rewritten_queries": ["machine learning definition", "ML overview"],
        "scope_status": "in_scope",
        "scope_reason": "",
        "retrieved_chunks": [],
        "raw_answer": "",
        "claim_verdicts": [],
        "validation_passed": False,
        "retry_count": 0,
        "final_answer": "",
        "sources": [],
        "status": "",
    }
    state.update(overrides)  # type: ignore
    return state


class TestRetrieveForQuery:
    """Tests for the _retrieve_for_query helper."""

    def test_returns_merged_chunk_ids(self):
        semantic_ids = ["c3", "c1", "c5"]
        bm25_ids = ["c1", "c4", "c6"]

        with patch.object(retriever_module, "embed_query", return_value=[0.1, 0.2, 0.3]) as mock_embed:
            with patch.object(retriever_module, "query_collection", return_value=semantic_ids) as mock_vec:
                with patch.object(retriever_module, "load_bm25", return_value=(MagicMock(), ["c1", "c2", "c3", "c4", "c5", "c6"])) as mock_load:
                    with patch.object(retriever_module, "bm25_search", return_value=bm25_ids) as mock_bm25:
                        result = _retrieve_for_query("test query", "doc123")

        mock_embed.assert_called_once_with("test query")
        mock_vec.assert_called_once_with("doc123", [0.1, 0.2, 0.3], n_results=15)
        mock_load.assert_called_once_with("doc123")
        mock_bm25.assert_called_once()
        assert isinstance(result, list)
        assert len(result) > 0
        for cid in result:
            assert isinstance(cid, str)


class TestRun:
    """Tests for the run() entry point."""

    def test_returns_top_5_chunks_as_dicts(self):
        chunk_dicts = [
            {"id": "c3", "doc_id": "doc123", "text": "ML is a subset of AI.", "page": 1, "chunk_type": "text", "section": "Intro"},
            {"id": "c1", "doc_id": "doc123", "text": "Deep learning uses neural networks.", "page": 2, "chunk_type": "text", "section": "Methods"},
            {"id": "c4", "doc_id": "doc123", "text": "Supervised learning requires labels.", "page": 3, "chunk_type": "text", "section": "Methods"},
            {"id": "c2", "doc_id": "doc123", "text": "Table of algorithms.", "page": 4, "chunk_type": "table", "section": "Appendix"},
            {"id": "c5", "doc_id": "doc123", "text": "Figure: AI pipeline diagram.", "page": 5, "chunk_type": "image_caption", "section": "Figures"},
        ]

        with patch.object(retriever_module, "_retrieve_for_query") as mock_retrieve:
            mock_retrieve.side_effect = [
                ["c1", "c3"],
                ["c2", "c4", "c5"],
            ]
            with patch.object(retriever_module, "get_chunks_by_ids", return_value=chunk_dicts):
                result = run(_make_state())

        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) == 5
        for c in result["retrieved_chunks"]:
            assert "id" in c
            assert "text" in c
            assert "page" in c
            assert "chunk_type" in c

    def test_returns_empty_on_error(self):
        with patch.object(retriever_module, "_retrieve_for_query", side_effect=RuntimeError("BM25 index missing")):
            result = run(_make_state())

        assert result == {"retrieved_chunks": []}

    def test_single_query_still_works(self):
        state = _make_state(rewritten_queries=["single query"])
        chunk_dicts = [
            {"id": "c1", "doc_id": "doc123", "text": "Text.", "page": 1, "chunk_type": "text", "section": "S1"},
        ]

        with patch.object(retriever_module, "_retrieve_for_query") as mock_retrieve:
            mock_retrieve.return_value = ["c1"]
            with patch.object(retriever_module, "get_chunks_by_ids", return_value=chunk_dicts):
                result = run(state)

        assert len(result["retrieved_chunks"]) == 1
        mock_retrieve.assert_called_once_with("single query", "doc123")
