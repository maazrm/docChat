# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

VMARP (Validated Multi-Agent RAG Pipeline) is a local-first document Q&A system. Users upload PDFs (digital or scanned), ask questions, and get grounded, validated answers with page citations. A hallucination validator audits every claim against source chunks, and a re-generate loop retries up to 3 times on failure.

## Tech stack

| Layer | Tool |
|---|---|
| LLM + validation | `gpt-4o-mini` (OpenAI) |
| Embeddings | `text-embedding-3-small` (OpenAI) |
| Document parsing | `docling` (IBM) — digital + scanned PDFs, tables, images |
| Orchestration | `langgraph` StateGraph |
| Vector store | `chromadb` (local persistent, no server) |
| Keyword search | `rank_bm25` |
| Hybrid merge | Reciprocal Rank Fusion (`core/rrf.py`) |
| Document store | `sqlite3` (stdlib) |
| UI | `streamlit` |
| Config | `python-dotenv` |

No system dependencies. No servers. Runs fully locally except for OpenAI API calls.

## Architecture — agent pipeline

Seven agents wired into a LangGraph `StateGraph` (`pipeline/graph.py`). Agents share state via `PipelineState` TypedDict (`pipeline/state.py`). Every agent function has signature `def run(state: PipelineState) -> dict` and returns only the keys it modifies.

```
                    ┌─────────────┐
                    │  scope_guard │  Agent 2: classify query in/out of scope
                    └──────┬──────┘
                           │ out_of_scope → END
                           ▼
                    ┌──────────────┐
                    │query_rewriter│  Agent 3: expand into 1–3 search queries
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  retriever   │  Agent 4: BM25 + semantic hybrid search → RRF merge
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  generator   │  Agent 5: synthesize grounded answer with page citations
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  validator   │  Agent 6: LLM-as-judge audit of every claim
                    └──────┬───────┘
                           │ pass → END
                           │ fail + retries < 3 → regenerator
                           │ fail + retries ≥ 3 → END (fallback)
                           ▼
                    ┌──────────────┐
                    │ regenerator  │  Agent 7: re-generate with failure context → back to validator
                    └──────────────┘
```

Agent 1 (preprocessor) runs **outside** the graph — called once when a PDF is uploaded, before any query loop. It parses → chunks → captions images → embeds → indexes (Chroma + BM25 + SQLite).

## Directory structure

```
core/           — reusable utilities (no LangGraph, no Streamlit)
  embedder.py   — OpenAI text-embedding-3-small wrapper
  vector_store.py — ChromaDB persistence (per-doc collections)
  bm25_index.py — build/save/load/search BM25 keyword index
  document_store.py — SQLite for full chunk text + metadata
  rrf.py        — Reciprocal Rank Fusion for hybrid merge

ingestion/      — PDF processing (called by preprocessor agent)
  docling_parser.py — Docling PDF → DoclingDocument + markdown export
  image_captioner.py — GPT-4o-mini vision captions for figures
  chunker.py    — Docling HybridChunker (structure-aware, 500 token chunks)

pipeline/
  state.py      — PipelineState, Chunk, ClaimVerdict TypedDicts
  graph.py      — LangGraph StateGraph wiring + conditional routing
  agents/
    preprocessor.py  — Agent 1: ingest PDF, populate all indexes
    scope_guard.py   — Agent 2: classify query scope
    query_rewriter.py — Agent 3: expand query into sub-queries
    retriever.py     — Agent 4: hybrid BM25 + semantic search
    generator.py     — Agent 5: grounded answer synthesis
    validator.py     — Agent 6: hallucination audit
    regenerator.py   — Agent 7: re-generate with failure context

app.py          — Streamlit UI entry point
```

## Running the app

```bash
streamlit run app.py
```

## Running tests

```bash
python -m pytest tests/ -v
```

Single test file:

```bash
python -m pytest tests/test_preprocessor.py -v
```

## Key conventions

- All LLM calls use `temperature=0` for deterministic output.
- Every agent wraps its logic in `try/except` — agents must never crash silently. On error, fail open (continue pipeline) or return a fallback response.
- ChromaDB creates one collection per document (keyed by `doc_id` MD5 hash) to prevent cross-document pollution. Collections are named `vmarp_{doc_id}`.
- BM25 indexes are serialized to `data/bm25_index_{doc_id}.pkl` (one per document).
- Docling auto-downloads layout models (~1–2 GB) to `~/.cache/docling/` on first use. This is a one-time download.

## Current state

The project scaffold is complete (folders, `__init__.py` files, `requirements.txt`, `.env.example`, `.gitignore`). All source files are empty stubs. The full implementation spec is in `VMARP_SPEC.md`, organized into 9 numbered sections meant to be built sequentially — each section depends on the previous ones and maps to one implementation session.

## Implementation order (from VMARP_SPEC.md)

| Section | What gets built |
|---|---|
| 01 | Project setup — folder structure, config files, requirements |
| 02 | Core utilities — embedder, vector store, BM25, SQLite, RRF |
| 03 | Document ingestion — Docling parser, image captioner, chunker |
| 04 | Pipeline state — shared TypedDicts used by all agents |
| 05 | Agents 1–3 — preprocessor, scope guard, query rewriter |
| 06 | Agents 4–5 — retriever, generator |
| 07 | Agents 6–7 — hallucination validator, re-generate agent |
| 08 | LangGraph graph — wires all agents into the state machine |
| 09 | Streamlit UI — the frontend that ties everything together |
