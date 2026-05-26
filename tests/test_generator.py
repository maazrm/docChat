import pytest
from unittest.mock import patch, MagicMock

from pipeline.agents import generator as generator_module
from pipeline.agents.generator import run, _format_chunks
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
        assert output.index("[Chunk 1") < output.index("[Chunk 2")

    def test_handles_image_caption_chunks(self):
        chunks = [
            {"id": "c1", "doc_id": "d1", "text": "Figure 1: A diagram.", "page": 2, "chunk_type": "image_caption", "section": "Figures"},
        ]
        output = _format_chunks(chunks)

        assert "[Chunk 1 | Page 2 | Type: image_caption]" in output
        assert "Figure 1: A diagram." in output

    def test_empty_chunks_returns_empty_string(self):
        assert _format_chunks([]) == ""


class TestRun:
    """Tests for the run() entry point."""

    def test_returns_raw_answer_and_sources(self):
        mock_answer = "Machine learning is a subset of AI (Page 1). Deep learning uses neural networks (Page 2)."

        with patch.object(generator_module, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = mock_answer
            mock_client.chat.completions.create.return_value = mock_response

            result = run(_make_state())

        assert result["raw_answer"] == mock_answer
        assert "sources" in result
        assert len(result["sources"]) == 2
        assert result["sources"][0]["page"] == 1
        assert result["sources"][1]["page"] == 2
        assert "text_snippet" in result["sources"][0]

    def test_calls_llm_with_formatted_context(self):
        state = _make_state()

        with patch.object(generator_module, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Answer."
            mock_client.chat.completions.create.return_value = mock_response

            run(state)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0
        prompt = call_kwargs["messages"][1]["content"]
        assert "What is machine learning?" in prompt
        assert "Chunk 1" in prompt
        assert "Chunk 2" in prompt

    def test_returns_fallback_on_error(self):
        with patch.object(generator_module, "client") as mock_client:
            mock_client.chat.completions.create.side_effect = RuntimeError("API error")

            result = run(_make_state())

        assert "error" in result["raw_answer"].lower()
        assert result["sources"] == []

    def test_sources_include_text_snippets(self):
        mock_answer = "Test answer."

        with patch.object(generator_module, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = mock_answer
            mock_client.chat.completions.create.return_value = mock_response

            result = run(_make_state())

        snippet = result["sources"][0]["text_snippet"]
        assert snippet.endswith("...")
        assert len(snippet) <= 153  # 150 chars + "..."
