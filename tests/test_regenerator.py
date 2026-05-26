import pytest
from unittest.mock import patch, MagicMock

from pipeline.agents import regenerator as regenerator_module
from pipeline.agents.regenerator import run, _format_chunks, MAX_RETRIES
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
        "raw_answer": "AI was invented in 1950. Deep learning uses neural networks (Page 2).",
        "claim_verdicts": [
            {"claim": "AI was invented in 1950", "verdict": "unsupported", "reason": "Not in any chunk."},
            {"claim": "Deep learning uses neural networks", "verdict": "supported", "reason": "Found in chunk 2."},
        ],
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

    def test_retry_increments_count(self):
        with patch.object(regenerator_module, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Corrected answer (Page 1)."
            mock_client.chat.completions.create.return_value = mock_response

            result = run(_make_state(retry_count=0))

        assert result["retry_count"] == 1

    def test_max_retries_returns_fallback(self):
        result = run(_make_state(retry_count=3))

        assert result["retry_count"] == 4
        assert result["validation_passed"] is True
        assert result["status"] == "fallback"
        assert "unable to generate" in result["final_answer"].lower()
        assert "Page 1" in result["final_answer"]
        assert "Page 2" in result["final_answer"]

    def test_injects_unsupported_claims_into_prompt(self):
        with patch.object(regenerator_module, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Corrected answer (Page 1)."
            mock_client.chat.completions.create.return_value = mock_response

            run(_make_state(retry_count=0))

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0
        prompt = call_kwargs["messages"][1]["content"]
        assert "AI was invented in 1950" in prompt
        assert "unsupported or ungrounded" in prompt.lower()
        assert "What is machine learning?" in prompt
        assert "Chunk 1" in prompt

    def test_returns_new_raw_answer(self):
        corrected = "Machine learning is a subset of AI (Page 1)."

        with patch.object(regenerator_module, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = corrected
            mock_client.chat.completions.create.return_value = mock_response

            result = run(_make_state(retry_count=1))

        assert result["raw_answer"] == corrected
        assert result["retry_count"] == 2

    def test_error_returns_fallback(self):
        with patch.object(regenerator_module, "client") as mock_client:
            mock_client.chat.completions.create.side_effect = RuntimeError("API error")

            result = run(_make_state(retry_count=0))

        assert result["retry_count"] == 1
        assert result["validation_passed"] is True
        assert result["status"] == "fallback"
        assert "unable to generate" in result["final_answer"].lower()

    def test_max_retries_boundary(self):
        """retry_count=4 already exceeds MAX_RETRIES=3."""
        result = run(_make_state(retry_count=4))
        assert result["validation_passed"] is True
        assert result["status"] == "fallback"

    def test_no_claim_verdicts_still_regenerates(self):
        """When claim_verdicts is empty, unsupported list is empty but regeneration still runs."""
        with patch.object(regenerator_module, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Regenerated answer (Page 1)."
            mock_client.chat.completions.create.return_value = mock_response

            result = run(_make_state(claim_verdicts=[], retry_count=0))

        assert result["raw_answer"] == "Regenerated answer (Page 1)."
        assert result["retry_count"] == 1
