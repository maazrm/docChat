import json
import pytest
from unittest.mock import patch, MagicMock

from pipeline.agents import validator as validator_module
from pipeline.agents.validator import run, _format_chunks
from pipeline.state import PipelineState


def _make_state(**overrides) -> PipelineState:
    state: PipelineState = {
        "doc_id": "doc123",
        "topic_summary": "Test document about AI.",
        "raw_query": "What is machine learning?",
        "rewritten_queries": ["machine learning definition"],
        "scope_status": "in_scope",
        "scope_reason": "",
        "retrieved_chunks": [
            {"id": "c1", "doc_id": "doc123", "text": "Machine learning is a subset of artificial intelligence.", "page": 1, "chunk_type": "text", "section": "Intro"},
            {"id": "c2", "doc_id": "doc123", "text": "Deep learning uses multi-layer neural networks.", "page": 2, "chunk_type": "text", "section": "Methods"},
        ],
        "raw_answer": "Machine learning is a subset of AI (Page 1). Deep learning uses neural networks (Page 2).",
        "claim_verdicts": [],
        "validation_passed": False,
        "retry_count": 0,
        "final_answer": "",
        "sources": [],
        "status": "",
    }
    state.update(overrides)  # type: ignore
    return state


class TestFormatChunks:
    """Tests for the _format_chunks helper."""

    def test_formats_chunks_with_page_and_type(self):
        chunks = [
            {"id": "c1", "doc_id": "d1", "text": "Hello world.", "page": 1, "chunk_type": "text", "section": "Intro"},
            {"id": "c2", "doc_id": "d1", "text": "A table of data.", "page": 3, "chunk_type": "table", "section": "Data"},
        ]
        output = _format_chunks(chunks)

        assert "[Chunk 1 | Page 1 | Type: text]" in output
        assert "Hello world." in output
        assert "[Chunk 2 | Page 3 | Type: table]" in output
        assert "A table of data." in output

    def test_empty_chunks_returns_empty_string(self):
        assert _format_chunks([]) == ""


class TestRun:
    """Tests for the run() entry point."""

    def test_passed_returns_success(self):
        verdict_data = {
            "all_supported": True,
            "verdicts": [
                {"claim": "ML is a subset of AI", "verdict": "supported", "reason": "Found in chunk 1."},
                {"claim": "Deep learning uses neural networks", "verdict": "supported", "reason": "Found in chunk 2."},
            ]
        }

        with patch.object(validator_module, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(verdict_data)
            mock_client.chat.completions.create.return_value = mock_response

            result = run(_make_state())

        assert result["validation_passed"] is True
        assert result["final_answer"] == "Machine learning is a subset of AI (Page 1). Deep learning uses neural networks (Page 2)."
        assert result["status"] == "success"
        assert len(result["claim_verdicts"]) == 2

    def test_failed_returns_unsupported(self):
        verdict_data = {
            "all_supported": False,
            "verdicts": [
                {"claim": "ML is a subset of AI", "verdict": "supported", "reason": "Found in chunk 1."},
                {"claim": "AI was invented in 1950", "verdict": "unsupported", "reason": "Not present in any chunk."},
            ]
        }

        with patch.object(validator_module, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(verdict_data)
            mock_client.chat.completions.create.return_value = mock_response

            result = run(_make_state())

        assert result["validation_passed"] is False
        assert len(result["claim_verdicts"]) == 2
        assert "final_answer" not in result

    def test_error_passes_through(self):
        with patch.object(validator_module, "client") as mock_client:
            mock_client.chat.completions.create.side_effect = RuntimeError("API error")

            result = run(_make_state())

        assert result["validation_passed"] is True
        assert result["claim_verdicts"] == []
        assert result["final_answer"] == "Machine learning is a subset of AI (Page 1). Deep learning uses neural networks (Page 2)."
        assert result["status"] == "success"

    def test_llm_call_format(self):
        state = _make_state()

        with patch.object(validator_module, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps({"all_supported": True, "verdicts": []})
            mock_client.chat.completions.create.return_value = mock_response

            run(state)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0
        prompt = call_kwargs["messages"][0]["content"]
        assert "Machine learning is a subset of AI" in prompt
        assert "Chunk 1" in prompt
        assert "Chunk 2" in prompt

    def test_missing_all_supported_defaults_to_false(self):
        verdict_data = {
            "verdicts": [
                {"claim": "Some claim", "verdict": "unsupported", "reason": "No evidence."},
            ]
        }

        with patch.object(validator_module, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(verdict_data)
            mock_client.chat.completions.create.return_value = mock_response

            result = run(_make_state())

        assert result["validation_passed"] is False
