import pytest
from unittest.mock import patch

from pipeline.graph import build_graph, _route_after_scope, _route_after_validation
from pipeline.state import PipelineState
from langgraph.graph import END


def _make_state(**overrides) -> PipelineState:
    state: PipelineState = {
        "doc_id": "doc123",
        "topic_summary": "Test document about AI.",
        "raw_query": "What is machine learning?",
        "rewritten_queries": [],
        "scope_status": "",
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


class TestBuildGraph:
    def test_build_graph_compiles(self):
        graph = build_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")


class TestRouteAfterScope:
    def test_out_of_scope_returns_end(self):
        state = _make_state(scope_status="out_of_scope")
        assert _route_after_scope(state) == END

    def test_in_scope_returns_query_rewriter(self):
        state = _make_state(scope_status="in_scope")
        assert _route_after_scope(state) == "query_rewriter"


class TestRouteAfterValidation:
    def test_validation_passed_returns_end(self):
        state = _make_state(validation_passed=True)
        assert _route_after_validation(state) == END

    def test_max_retries_returns_end(self):
        state = _make_state(validation_passed=False, retry_count=3)
        assert _route_after_validation(state) == END

    def test_failed_with_retries_remaining_returns_regenerator(self):
        state = _make_state(validation_passed=False, retry_count=1)
        assert _route_after_validation(state) == "regenerator"

    def test_max_retries_exceeded_returns_end(self):
        state = _make_state(validation_passed=False, retry_count=5)
        assert _route_after_validation(state) == END


class TestGraphRouting:
    def test_out_of_scope_routes_to_end(self):
        state = _make_state(scope_status="out_of_scope")
        import pipeline.graph as graph_module

        with patch.object(graph_module.scope_guard, "run", return_value={
            "scope_status": "out_of_scope",
            "scope_reason": "Not about the document.",
            "final_answer": "Your question is out of scope.",
            "status": "out_of_scope",
        }):
            graph = build_graph()
            result = graph.invoke(state)

        assert result["status"] == "out_of_scope"
        assert result["raw_answer"] == ""

    def test_in_scope_runs_full_pipeline(self):
        state = _make_state(scope_status="in_scope")
        import pipeline.graph as graph_module

        with patch.object(graph_module.scope_guard, "run", return_value={
            "scope_status": "in_scope", "scope_reason": "Valid question."
        }), patch.object(graph_module.query_rewriter, "run", return_value={
            "rewritten_queries": ["ML definition"]
        }), patch.object(graph_module.retriever, "run", return_value={
            "retrieved_chunks": [
                {"id": "c1", "doc_id": "doc123", "text": "ML is AI.", "page": 1,
                 "chunk_type": "text", "section": "Intro"}
            ]
        }), patch.object(graph_module.generator, "run", return_value={
            "raw_answer": "ML is AI (Page 1).",
            "sources": [{"page": 1, "chunk_type": "text", "text_snippet": "ML is AI."}]
        }), patch.object(graph_module.validator, "run", return_value={
            "validation_passed": True,
            "claim_verdicts": [
                {"claim": "ML is AI.", "verdict": "supported", "reason": "Found."}
            ],
            "final_answer": "ML is AI (Page 1).",
            "status": "success",
        }):
            graph = build_graph()
            result = graph.invoke(state)

        assert result["status"] == "success"
        assert result["validation_passed"] is True

    def test_validation_retry_loop(self):
        state = _make_state(
            scope_status="in_scope",
            raw_answer="Initial answer.",
            retry_count=0,
        )
        import pipeline.graph as graph_module

        validator_responses = [
            {"validation_passed": False, "claim_verdicts": [
                {"claim": "X.", "verdict": "unsupported", "reason": "Not found."}
            ]},
            {"validation_passed": True, "claim_verdicts": [
                {"claim": "Y.", "verdict": "supported", "reason": "Found."}
            ], "final_answer": "Corrected answer.", "status": "success"},
        ]

        with patch.object(graph_module.scope_guard, "run", return_value={
            "scope_status": "in_scope", "scope_reason": "Valid."
        }), patch.object(graph_module.query_rewriter, "run", return_value={
            "rewritten_queries": ["Q1"]
        }), patch.object(graph_module.retriever, "run", return_value={
            "retrieved_chunks": [
                {"id": "c1", "doc_id": "doc123", "text": "Content.", "page": 1,
                 "chunk_type": "text", "section": "S1"}
            ]
        }), patch.object(graph_module.generator, "run", return_value={
            "raw_answer": "Initial answer.", "sources": []
        }), patch.object(graph_module.validator, "run", side_effect=validator_responses), \
        patch.object(graph_module.regenerator, "run", return_value={
            "raw_answer": "Corrected answer.", "retry_count": 1
        }):
            graph = build_graph()
            result = graph.invoke(state)

        assert result["validation_passed"] is True
        assert result["final_answer"] == "Corrected answer."

    def test_max_retries_ends_with_fallback(self):
        state = _make_state(
            scope_status="in_scope",
            retry_count=3,
        )
        import pipeline.graph as graph_module

        with patch.object(graph_module.scope_guard, "run", return_value={
            "scope_status": "in_scope", "scope_reason": "Valid."
        }), patch.object(graph_module.query_rewriter, "run", return_value={
            "rewritten_queries": ["Q1"]
        }), patch.object(graph_module.retriever, "run", return_value={
            "retrieved_chunks": [
                {"id": "c1", "doc_id": "doc123", "text": "Content.", "page": 1,
                 "chunk_type": "text", "section": "S1"}
            ]
        }), patch.object(graph_module.generator, "run", return_value={
            "raw_answer": "Answer.", "sources": []
        }), patch.object(graph_module.validator, "run", return_value={
            "validation_passed": False,
            "claim_verdicts": [
                {"claim": "X.", "verdict": "unsupported", "reason": "Not found."}
            ],
        }):
            graph = build_graph()
            result = graph.invoke(state)

        assert result["validation_passed"] is False
