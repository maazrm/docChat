from typing import TypedDict, Optional


class Chunk(TypedDict):
    """Represents a single chunk of document content."""
    id:         str            # Unique ID: "doc_<hash>_chunk_<n>"
    text:       str            # The chunk's text content
    page:       int            # Page number in the source document
    chunk_type: str            # "text" | "table" | "image_caption"
    section:    Optional[str]  # Nearest heading above this chunk, if any
    doc_id:     str            # MD5 hash of the source document


class ClaimVerdict(TypedDict):
    """Result of validating a single claim from the generator's answer."""
    claim:   str  # The extracted claim text
    verdict: str  # "supported" | "unsupported"
    reason:  str  # One-sentence explanation from the validator


class PipelineState(TypedDict):
    """
    The full state object that flows through the LangGraph pipeline.
    Each agent receives this state and returns a dict of only the keys it modifies.
    """

    # --- Set during ingestion (before the graph runs) ---
    doc_id:        str  # MD5 hash of the uploaded PDF
    topic_summary: str  # 2-3 sentence summary generated from the document

    # --- Set at query time ---
    raw_query:        str        # The user's original question
    rewritten_queries: list[str] # Expanded queries from Agent 3

    # --- Set by scope guard (Agent 2) ---
    scope_status: str  # "in_scope" | "out_of_scope"
    scope_reason: str  # One-sentence explanation

    # --- Set by retriever (Agent 4) ---
    retrieved_chunks: list[Chunk]  # Top 5 chunks after hybrid search + RRF

    # --- Set by generator (Agent 5) ---
    raw_answer: str  # Answer before validation

    # --- Set by validator (Agent 6) and regenerator (Agent 7) ---
    claim_verdicts:   list[ClaimVerdict]
    validation_passed: bool
    retry_count:       int  # Number of regeneration attempts so far (starts at 0)

    # --- Final output ---
    final_answer: str
    sources:      list[dict]  # [{page, chunk_type, text_snippet}] for UI display
    status:       str         # "success" | "fallback" | "out_of_scope"
