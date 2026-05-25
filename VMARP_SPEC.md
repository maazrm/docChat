# VMARP — Validated Multi-Agent RAG Pipeline
### Full Project Spec — Split by Implementation Section

---

## How to use this file

This spec is divided into **9 numbered sections**. Each section is self-contained and maps to one Claude Code session. Work through them in order — each section depends on what was built before it.

Copy each section into its own file (e.g. `SPEC_01_PROJECT_SETUP.md`) and hand it to Claude Code one at a time.



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

---

---

# SECTION 01 — Project Setup

## Goal
Create the full project folder structure, install dependencies, and set up config files. No logic yet — just scaffolding.

## What to build

### Folder structure
Create every folder and empty `__init__.py` file as shown:

```
vmarp/
├── app.py                        # empty for now
├── requirements.txt
├── .env                          # not committed
├── .env.example
├── .gitignore
├── README.md                     # empty for now
│
├── pipeline/
│   ├── __init__.py
│   ├── graph.py                  # empty for now
│   ├── state.py                  # empty for now
│   └── agents/
│       ├── __init__.py
│       ├── preprocessor.py       # empty for now
│       ├── scope_guard.py        # empty for now
│       ├── query_rewriter.py     # empty for now
│       ├── retriever.py          # empty for now
│       ├── generator.py          # empty for now
│       ├── validator.py          # empty for now
│       └── regenerator.py        # empty for now
│
├── core/
│   ├── __init__.py
│   ├── embedder.py               # empty for now
│   ├── vector_store.py           # empty for now
│   ├── bm25_index.py             # empty for now
│   ├── document_store.py         # empty for now
│   └── rrf.py                    # empty for now
│
├── ingestion/
│   ├── __init__.py
│   ├── docling_parser.py         # empty for now
│   ├── image_captioner.py        # empty for now
│   └── chunker.py                # empty for now
│
├── data/
│   └── .gitkeep
│
└── tests/
    ├── test_preprocessor.py      # empty for now
    ├── test_retriever.py         # empty for now
    ├── test_validator.py         # empty for now
    └── sample_docs/              # drop a small test PDF here manually
```

---

### `requirements.txt`

```
# LLM + Embeddings
openai>=1.30.0

# Orchestration
langgraph>=0.1.0
langchain>=0.2.0
langchain-openai>=0.1.0

# Document parsing (handles digital + scanned PDFs, tables, images — no system deps needed)
docling>=2.0.0

# Vector store
chromadb>=0.5.0

# Keyword search
rank_bm25>=0.2.2

# Utilities
tiktoken>=0.7.0
streamlit>=1.35.0
python-dotenv>=1.0.0
Pillow>=10.0.0
```

---

### `.env.example`

```
OPENAI_API_KEY=sk-...
```

---

### `.gitignore`

```
.env
data/chroma_db/
data/bm25_index.pkl
data/documents.db
__pycache__/
*.pyc
.venv/
uploads/
```

---

## Deliverable checklist
- [ ] All folders created
- [ ] All `__init__.py` files present
- [ ] `requirements.txt` written
- [ ] `.env.example` written
- [ ] `.gitignore` written
- [ ] `pip install -r requirements.txt` runs without errors

---

---

# SECTION 02 — Core Utilities

## Goal
Build the five utility modules that every agent in the pipeline depends on. No agent logic yet — just reusable building blocks.

## Dependencies from previous sections
- Section 01 complete (folder structure and packages installed)
- `OPENAI_API_KEY` set in `.env`

## Context
These five files live in `core/`. They are pure utility — no LangGraph, no Streamlit. Each one does one thing and exposes a clean function interface that agents will import.

---

## What to build

### `core/embedder.py`
Wraps the OpenAI `text-embedding-3-small` model. Used during ingestion (to embed all chunks) and at query time (to embed the user's query).

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using text-embedding-3-small.
    Processes in batches of 100 to stay within API limits.
    """
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        all_embeddings.extend([r.embedding for r in response.data])
    return all_embeddings

def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
```

---

### `core/vector_store.py`
Wraps ChromaDB. Handles creating a collection, upserting chunks, and querying by vector similarity. ChromaDB persists to `data/chroma_db/` on disk — no server needed.

```python
import chromadb
from chromadb.config import Settings

CLIENT = chromadb.PersistentClient(path="data/chroma_db")

def get_or_create_collection(doc_id: str):
    """
    Each document gets its own Chroma collection, keyed by doc_id.
    This prevents documents from polluting each other's indexes.
    """
    return CLIENT.get_or_create_collection(
        name=f"vmarp_{doc_id}",
        metadata={"hnsw:space": "cosine"}
    )

def upsert_chunks(doc_id: str, chunks: list[dict], embeddings: list[list[float]]):
    """
    Upsert chunks and their embeddings into the collection.
    chunks is a list of Chunk dicts (see pipeline/state.py).
    embeddings is a parallel list of embedding vectors.
    """
    collection = get_or_create_collection(doc_id)
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "page":       c["page"],
            "chunk_type": c["chunk_type"],
            "section":    c.get("section") or "",
            "doc_id":     c["doc_id"]
        } for c in chunks]
    )

def query_collection(doc_id: str, query_embedding: list[float], n_results: int = 15) -> list[str]:
    """
    Query the collection by vector similarity.
    Returns a ranked list of chunk IDs (most similar first).
    """
    collection = get_or_create_collection(doc_id)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["ids"][0]  # list of chunk IDs
```

---

### `core/bm25_index.py`
Builds and persists a BM25 keyword index from chunk texts. Used alongside vector search for hybrid retrieval. Index is serialized to `data/bm25_index_{doc_id}.pkl` so it survives app restarts.

```python
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

def _index_path(doc_id: str) -> Path:
    return Path(f"data/bm25_index_{doc_id}.pkl")

def build_and_save(doc_id: str, chunks: list[dict]) -> None:
    """
    Build a BM25 index from chunk texts and save it to disk.
    Call this once during ingestion.
    """
    corpus = [chunk["text"].lower().split() for chunk in chunks]
    index = BM25Okapi(corpus)
    chunk_ids = [c["id"] for c in chunks]

    path = _index_path(doc_id)
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"index": index, "chunk_ids": chunk_ids}, f)

def load(doc_id: str) -> tuple[BM25Okapi, list[str]]:
    """Load a previously saved BM25 index from disk."""
    with open(_index_path(doc_id), "rb") as f:
        data = pickle.load(f)
    return data["index"], data["chunk_ids"]

def search(query: str, index: BM25Okapi, chunk_ids: list[str], n: int = 15) -> list[str]:
    """
    Run BM25 keyword search.
    Returns a ranked list of chunk IDs (highest score first).
    """
    tokens = query.lower().split()
    scores = index.get_scores(tokens)
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [chunk_ids[i] for i in top_n]

def index_exists(doc_id: str) -> bool:
    return _index_path(doc_id).exists()
```

---

### `core/document_store.py`
SQLite store for raw chunk data. Chroma stores vectors; SQLite stores the full chunk text and metadata so agents can fetch complete chunk data by ID after retrieval.

```python
import sqlite3
import json
from pathlib import Path

DB_PATH = Path("data/documents.db")

def _connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    """Create the chunks table if it doesn't exist. Call once at startup."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id         TEXT PRIMARY KEY,
                doc_id     TEXT NOT NULL,
                text       TEXT NOT NULL,
                page       INTEGER,
                chunk_type TEXT,
                section    TEXT
            )
        """)
        conn.commit()

def save_chunks(chunks: list[dict]):
    """Insert or replace a list of chunk dicts into the database."""
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (id, doc_id, text, page, chunk_type, section)
            VALUES (:id, :doc_id, :text, :page, :chunk_type, :section)
            """,
            chunks
        )
        conn.commit()

def get_chunks_by_ids(chunk_ids: list[str]) -> list[dict]:
    """Fetch full chunk data for a list of IDs. Returns in the same order."""
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT id, doc_id, text, page, chunk_type, section FROM chunks WHERE id IN ({placeholders})",
            chunk_ids
        ).fetchall()

    row_map = {
        row[0]: {
            "id": row[0], "doc_id": row[1], "text": row[2],
            "page": row[3], "chunk_type": row[4], "section": row[5]
        }
        for row in rows
    }
    # Return in the original order (rank order from retrieval)
    return [row_map[cid] for cid in chunk_ids if cid in row_map]

def doc_exists(doc_id: str) -> bool:
    """Check if chunks for a given doc_id are already stored."""
    with _connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
    return count > 0
```

---

### `core/rrf.py`
Reciprocal Rank Fusion — merges multiple ranked lists (one from BM25, one from vector search) into a single ranked list. This is the "hybrid" in hybrid search.

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60
) -> list[str]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Each document's score is the sum of 1/(k + rank) across all lists.
    k=60 is the standard default from the original RRF paper.

    Args:
        ranked_lists: e.g. [bm25_results, semantic_results] where each
                      is a list of chunk IDs sorted best-first.
        k:            smoothing constant (default 60).

    Returns:
        A single merged list of chunk IDs, best-first.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
```

---

## Deliverable checklist
- [ ] `core/embedder.py` — `embed_texts()` and `embed_query()` work
- [ ] `core/vector_store.py` — can upsert and query a Chroma collection
- [ ] `core/bm25_index.py` — can build, save, load, and search an index
- [ ] `core/document_store.py` — can init DB, save chunks, fetch by ID
- [ ] `core/rrf.py` — merges two ranked lists correctly
- [ ] Quick smoke test: call `embed_query("test")` and confirm a vector is returned

---

---

# SECTION 03 — Document Ingestion

## Goal
Build the three ingestion modules that take a raw PDF and produce a clean list of chunks ready for indexing. Uses Docling for all parsing — no separate OCR setup needed.

## Dependencies from previous sections
- Section 01 complete
- Section 02 complete (`core/embedder.py` available for import)
- `OPENAI_API_KEY` set in `.env`

## Context
These three files live in `ingestion/`. They are called by the preprocessor agent (Section 05) but are built here as standalone, testable modules.

**Important about Docling:** On the very first call to `DocumentConverter().convert()`, Docling will automatically download its AI layout models (~1–2 GB). This is a one-time download that caches to `~/.cache/docling/`. Subsequent runs are fast. No manual setup needed.

---

## What to build

### `ingestion/docling_parser.py`
Converts a PDF (digital or scanned) into a Docling `DoclingDocument` object. Docling auto-detects whether OCR is needed — no branching logic required.

```python
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

def parse_pdf(file_path: str):
    """
    Convert a PDF to a Docling document object.

    Docling handles both digital and scanned PDFs automatically.
    For scanned docs it runs OCR internally — no Tesseract setup needed.
    Returns a DoclingDocument with .tables, .pictures, and export methods.
    """
    logger.info(f"Parsing PDF: {file_path}")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    logger.info(f"Parsing complete. Pages: {len(result.document.pages)}")
    return result.document

def get_markdown(document) -> str:
    """
    Export the full document as clean markdown.
    Tables are rendered as markdown tables. Headings are preserved.
    Used to generate the topic summary.
    """
    return document.export_to_markdown()
```

---

### `ingestion/image_captioner.py`
Sends figures extracted by Docling to `gpt-4o-mini` (vision) and returns a text caption for each. The caption becomes a chunk in the index so image content is searchable.

```python
import base64
import logging
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)

def caption_figure(image_bytes: bytes) -> str:
    """
    Send an image to gpt-4o-mini vision and return a descriptive caption.
    The caption is stored as a chunk so image content is searchable.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe what this image or figure shows in 1-2 sentences. "
                            "Focus on any data, labels, chart values, or key information visible. "
                            "Be specific and factual."
                        )
                    }
                ]
            }]
        )
        caption = response.choices[0].message.content
        logger.info(f"Captioned figure: {caption[:80]}...")
        return caption
    except Exception as e:
        logger.warning(f"Image captioning failed: {e}")
        return "Figure: content could not be captioned."

def caption_all_figures(document, doc_id: str) -> list[dict]:
    """
    Extract all figures from a Docling document, caption each one,
    and return them as a list of Chunk dicts ready for indexing.
    """
    caption_chunks = []
    for i, picture in enumerate(document.pictures):
        try:
            # Export figure as PNG bytes
            img_bytes = picture.get_image(document).tobytes()
            caption = caption_figure(img_bytes)
            page_no = picture.prov[0].page_no if picture.prov else 0
            caption_chunks.append({
                "id":         f"{doc_id}_figure_{i}",
                "text":       caption,
                "page":       page_no,
                "chunk_type": "image_caption",
                "section":    None,
                "doc_id":     doc_id
            })
        except Exception as e:
            logger.warning(f"Skipping figure {i}: {e}")

    logger.info(f"Captioned {len(caption_chunks)} figures")
    return caption_chunks
```

---

### `ingestion/chunker.py`
Uses Docling's built-in `HybridChunker` to split the document into RAG-ready chunks. The HybridChunker respects document structure — it never splits a table across chunks and keeps headings with their content.

```python
import logging
from docling.chunking import HybridChunker

logger = logging.getLogger(__name__)

def chunk_document(document, doc_id: str) -> list[dict]:
    """
    Chunk a Docling document using HybridChunker.

    HybridChunker respects document structure:
    - Never splits a table across chunks
    - Keeps headings attached to their content
    - Merges short adjacent chunks (merge_peers=True)

    Returns a list of Chunk dicts (matching the Chunk TypedDict in pipeline/state.py).
    Does NOT include image caption chunks — those are added separately by image_captioner.py.
    """
    chunker = HybridChunker(max_tokens=500, merge_peers=True)
    raw_chunks = list(chunker.chunk(document))

    chunks = []
    for i, chunk in enumerate(raw_chunks):
        meta = chunk.meta

        # Extract page number from first doc item's provenance
        page_no = 0
        if meta.doc_items and meta.doc_items[0].prov:
            page_no = meta.doc_items[0].prov[0].page_no

        # Extract chunk type from the Docling label (text, table, figure, etc.)
        chunk_type = "text"
        if meta.doc_items:
            label = str(meta.doc_items[0].label)
            if "table" in label.lower():
                chunk_type = "table"

        # Extract nearest section heading
        section = meta.headings[0] if meta.headings else None

        chunks.append({
            "id":         f"{doc_id}_chunk_{i}",
            "text":       chunk.text,
            "page":       page_no,
            "chunk_type": chunk_type,
            "section":    section,
            "doc_id":     doc_id
        })

    logger.info(f"Produced {len(chunks)} chunks from document")
    return chunks
```

---

## Deliverable checklist
- [ ] `ingestion/docling_parser.py` — `parse_pdf()` returns a Docling document object
- [ ] `ingestion/image_captioner.py` — `caption_all_figures()` returns a list of caption chunks
- [ ] `ingestion/chunker.py` — `chunk_document()` returns a list of chunk dicts
- [ ] Quick smoke test: parse one of the PDFs in `tests/sample_docs/` and print the first 3 chunk texts

---

---

# SECTION 04 — Pipeline State

## Goal
Define the shared TypedDicts that flow through the entire LangGraph pipeline. Every agent reads from and writes to this shared state object.

## Dependencies from previous sections
- Section 01 complete (folder structure in place)

## Context
This is one file: `pipeline/state.py`. It has no logic — just type definitions. Build it before any agents so every subsequent section can import cleanly.

---

## What to build

### `pipeline/state.py`

```python
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
```

---

## Deliverable checklist
- [ ] `pipeline/state.py` written with all three TypedDicts
- [ ] File imports cleanly: `from pipeline.state import PipelineState, Chunk, ClaimVerdict`

---

---

# SECTION 05 — Agents 1, 2, 3 (Preprocessor · Scope Guard · Query Rewriter)

## Goal
Build the first three agents: the preprocessor that ingests documents, the scope guard that filters off-topic queries, and the query rewriter that improves retrieval.

## Dependencies from previous sections
- Section 01 complete
- Section 02 complete — `core/` utilities available
- Section 03 complete — `ingestion/` modules available
- Section 04 complete — `PipelineState` importable

## Context
Every agent function must follow this signature:

```python
def run(state: PipelineState) -> dict:
    ...
    return {"key": value}  # return ONLY the keys this agent modifies
```

All OpenAI calls use `temperature=0` for deterministic outputs. Every call is wrapped in `try/except` — agents must never crash silently.

---

## What to build

### `pipeline/agents/preprocessor.py` — Agent 1

This agent is called **outside** the LangGraph query graph — it runs once when a document is uploaded, before any query loop. It populates the vector store, BM25 index, and SQLite, then returns the `doc_id` and `topic_summary` that will be stored in Streamlit session state.

```python
import hashlib
import logging
import tempfile
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

from ingestion.docling_parser import parse_pdf, get_markdown
from ingestion.image_captioner import caption_all_figures
from ingestion.chunker import chunk_document
from core.embedder import embed_texts
from core.vector_store import upsert_chunks
from core.bm25_index import build_and_save, index_exists
from core.document_store import init_db, save_chunks, doc_exists

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)


def compute_doc_id(file_bytes: bytes) -> str:
    """MD5 hash of the file bytes — used as the unique document identifier."""
    return hashlib.md5(file_bytes).hexdigest()


def generate_topic_summary(markdown_text: str) -> str:
    """Send the first ~1500 tokens of the document to gpt-4o-mini for a summary."""
    # Truncate to roughly 1500 tokens (~6000 chars)
    truncated = markdown_text[:6000]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You summarize documents concisely."
                },
                {
                    "role": "user",
                    "content": (
                        "In 2-3 sentences, summarize what this document is about. "
                        "Be specific about its domain, subject matter, and scope.\n\n"
                        f"Document excerpt:\n{truncated}"
                    )
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Topic summary generation failed: {e}")
        return "Document summary unavailable."


def ingest(file_bytes: bytes, filename: str) -> dict:
    """
    Full ingestion pipeline for an uploaded PDF.

    Returns:
        {
            "doc_id": str,
            "topic_summary": str,
            "chunk_count": int
        }

    Skips re-ingestion if this doc_id already exists in the database.
    """
    doc_id = compute_doc_id(file_bytes)
    logger.info(f"Starting ingestion for doc_id={doc_id} ({filename})")

    # Skip if already ingested
    init_db()
    if doc_exists(doc_id) and index_exists(doc_id):
        logger.info("Document already ingested — skipping.")
        # We still need the topic summary — re-generate it
        # (in a real app you'd cache this too, but for local use this is fine)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        document = parse_pdf(tmp_path)
        markdown = get_markdown(document)
        topic_summary = generate_topic_summary(markdown)
        return {"doc_id": doc_id, "topic_summary": topic_summary, "chunk_count": 0}

    # Save to a temp file for Docling to parse
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Step 1: Parse the PDF with Docling
    logger.info("Parsing PDF with Docling...")
    document = parse_pdf(tmp_path)
    markdown = get_markdown(document)

    # Step 2: Generate topic summary
    logger.info("Generating topic summary...")
    topic_summary = generate_topic_summary(markdown)

    # Step 3: Chunk the document
    logger.info("Chunking document...")
    text_chunks = chunk_document(document, doc_id)

    # Step 4: Caption any figures
    logger.info("Captioning figures...")
    figure_chunks = caption_all_figures(document, doc_id)

    all_chunks = text_chunks + figure_chunks
    logger.info(f"Total chunks: {len(all_chunks)}")

    # Step 5: Embed all chunks
    logger.info("Embedding chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    # Step 6: Save to all indexes
    logger.info("Saving to vector store, BM25 index, and SQLite...")
    upsert_chunks(doc_id, all_chunks, embeddings)
    build_and_save(doc_id, all_chunks)
    save_chunks(all_chunks)

    logger.info("Ingestion complete.")
    Path(tmp_path).unlink(missing_ok=True)

    return {
        "doc_id":        doc_id,
        "topic_summary": topic_summary,
        "chunk_count":   len(all_chunks)
    }
```

---

### `pipeline/agents/scope_guard.py` — Agent 2

```python
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from pipeline.state import PipelineState

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)


def run(state: PipelineState) -> dict:
    """
    Classify whether the user's query is answerable from the uploaded document.
    Returns scope_status ("in_scope" | "out_of_scope") and scope_reason.
    Short-circuits the pipeline if out of scope.
    """
    logger.info(f"[ScopeGuard] Query: {state['raw_query']}")

    prompt = f"""You are a strict scope classifier. Your job is to decide if a user's question can be answered using the provided document.

Document summary: {state['topic_summary']}

User question: {state['raw_query']}

Can this question be answered from the document described above?
Reply with a JSON object: {{"in_scope": true/false, "reason": "<one sentence>"}}
Only reply with the JSON. Nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        in_scope = parsed.get("in_scope", False)
        reason = parsed.get("reason", "")

        if in_scope:
            logger.info(f"[ScopeGuard] IN SCOPE — {reason}")
            return {
                "scope_status": "in_scope",
                "scope_reason": reason
            }
        else:
            logger.info(f"[ScopeGuard] OUT OF SCOPE — {reason}")
            return {
                "scope_status":  "out_of_scope",
                "scope_reason":  reason,
                "final_answer":  (
                    f"This question is outside the scope of the uploaded document. "
                    f"The document covers: {state['topic_summary']}. "
                    f"Please ask something related to its content."
                ),
                "status": "out_of_scope"
            }

    except Exception as e:
        logger.error(f"[ScopeGuard] Error: {e}")
        return {
            "scope_status": "in_scope",  # fail open — let the pipeline continue
            "scope_reason": "Scope check failed; proceeding."
        }
```

---

### `pipeline/agents/query_rewriter.py` — Agent 3

```python
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from pipeline.state import PipelineState

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)


def run(state: PipelineState) -> dict:
    """
    Rewrite the user's query into 1-3 targeted search queries.
    Multi-part or ambiguous questions are split into sub-queries
    so retrieval has better coverage.
    """
    logger.info(f"[QueryRewriter] Rewriting: {state['raw_query']}")

    prompt = f"""You are a query optimization assistant for a document retrieval system.

Given the user's question, generate 1 to 3 search queries that would retrieve the most relevant passages from a document.
- If the question is simple and specific, return just 1 query.
- If the question is multi-part or ambiguous, return 2-3 targeted sub-queries.
- Expand abbreviations. Remove filler words.
- Do NOT answer the question. Only rewrite it for search.

Return a JSON array of strings. Example: ["query 1", "query 2"]
Only return the JSON array. Nothing else.

User question: {state['raw_query']}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        queries = json.loads(raw)

        if not isinstance(queries, list) or not queries:
            queries = [state["raw_query"]]

        logger.info(f"[QueryRewriter] Rewritten queries: {queries}")
        return {"rewritten_queries": queries}

    except Exception as e:
        logger.error(f"[QueryRewriter] Error: {e}. Falling back to raw query.")
        return {"rewritten_queries": [state["raw_query"]]}
```

---

## Deliverable checklist
- [ ] `pipeline/agents/preprocessor.py` — `ingest()` processes a PDF and returns `doc_id` and `topic_summary`
- [ ] `pipeline/agents/scope_guard.py` — `run()` returns correct `scope_status`
- [ ] `pipeline/agents/query_rewriter.py` — `run()` returns a list of rewritten queries
- [ ] Smoke test: call `ingest()` on a sample PDF and confirm chunks appear in SQLite

---

---

# SECTION 06 — Agents 4 & 5 (Retriever · Generator)

## Goal
Build the retriever that runs hybrid BM25 + semantic search and the generator that synthesises a grounded answer from the retrieved chunks.

## Dependencies from previous sections
- Sections 01–05 complete
- `core/embedder.py`, `core/vector_store.py`, `core/bm25_index.py`, `core/document_store.py`, `core/rrf.py` all available
- `pipeline/state.py` available

---

## What to build

### `pipeline/agents/retriever.py` — Agent 4

```python
import logging
from core.embedder import embed_query
from core.vector_store import query_collection
from core.bm25_index import load as load_bm25, search as bm25_search
from core.document_store import get_chunks_by_ids
from core.rrf import reciprocal_rank_fusion
from pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def _retrieve_for_query(query: str, doc_id: str, n: int = 15) -> list[str]:
    """
    Run both semantic and BM25 search for a single query string.
    Returns a merged, RRF-ranked list of chunk IDs.
    """
    # Semantic search
    q_embedding = embed_query(query)
    semantic_ids = query_collection(doc_id, q_embedding, n_results=n)

    # BM25 keyword search
    bm25_index, chunk_ids = load_bm25(doc_id)
    bm25_ids = bm25_search(query, bm25_index, chunk_ids, n=n)

    # Merge with RRF
    merged = reciprocal_rank_fusion([semantic_ids, bm25_ids])
    return merged


def run(state: PipelineState) -> dict:
    """
    Hybrid retrieval over all rewritten queries.
    Runs BM25 + semantic search for each query, merges all results with RRF,
    and returns the top 5 chunks as full Chunk dicts.
    """
    logger.info(f"[Retriever] Queries: {state['rewritten_queries']}")

    all_ranked_lists = []
    for query in state["rewritten_queries"]:
        ranked = _retrieve_for_query(query, state["doc_id"])
        all_ranked_lists.append(ranked)

    # Final RRF merge across all sub-query result lists
    final_ranking = reciprocal_rank_fusion(all_ranked_lists)
    top_ids = final_ranking[:5]

    chunks = get_chunks_by_ids(top_ids)
    logger.info(f"[Retriever] Retrieved {len(chunks)} chunks from pages: {[c['page'] for c in chunks]}")

    return {"retrieved_chunks": chunks}
```

---

### `pipeline/agents/generator.py` — Agent 5

```python
import logging
from openai import OpenAI
from dotenv import load_dotenv
from pipeline.state import PipelineState, Chunk

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)


def _format_chunks(chunks: list[Chunk]) -> str:
    """Format retrieved chunks into a readable context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | Page {chunk['page']} | Type: {chunk['chunk_type']}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def run(state: PipelineState) -> dict:
    """
    Generate a grounded answer from the retrieved chunks.
    Every claim must cite a page number. No speculation beyond the context.
    """
    logger.info(f"[Generator] Generating answer for: {state['raw_query']}")

    formatted_context = _format_chunks(state["retrieved_chunks"])

    prompt = f"""You are a precise document analyst. Answer the user's question using ONLY the provided context passages.

Rules:
- Base every claim strictly on the provided context.
- If the context does not contain enough information to fully answer, say so explicitly.
- Do NOT speculate or add information from outside the context.
- For each key claim, cite the page number in parentheses, e.g. (Page 4).
- Keep the answer clear and well-structured.

Context:
{formatted_context}

Question: {state['raw_query']}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a precise, grounded document analyst."},
                {"role": "user",   "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
        logger.info(f"[Generator] Answer: {answer[:120]}...")

        # Build sources list for the UI
        sources = [
            {
                "page":         c["page"],
                "chunk_type":   c["chunk_type"],
                "text_snippet": c["text"][:150] + "..."
            }
            for c in state["retrieved_chunks"]
        ]

        return {"raw_answer": answer, "sources": sources}

    except Exception as e:
        logger.error(f"[Generator] Error: {e}")
        return {
            "raw_answer": "An error occurred while generating the answer. Please try again.",
            "sources":    []
        }
```

---

## Deliverable checklist
- [ ] `pipeline/agents/retriever.py` — `run()` returns 5 chunk dicts
- [ ] `pipeline/agents/generator.py` — `run()` returns `raw_answer` and `sources`
- [ ] Manually test: call both agents with a pre-populated state dict and verify output

---

---

# SECTION 07 — Agents 6 & 7 (Validator · Re-Generate Agent)

## Goal
Build the hallucination validator that audits every claim in the generated answer, and the re-generate agent that retries generation with failure context injected.

## Dependencies from previous sections
- Sections 01–06 complete
- `pipeline/state.py` available

---

## What to build

### `pipeline/agents/validator.py` — Agent 6

```python
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from pipeline.state import PipelineState, Chunk, ClaimVerdict

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)


def _format_chunks(chunks: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | Page {chunk['page']} | Type: {chunk['chunk_type']}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def run(state: PipelineState) -> dict:
    """
    Audit every factual claim in the generated answer against the source chunks.

    Uses a second gpt-4o-mini call (LLM-as-judge) to verify grounding.
    A claim fails if it is not mentioned in the context, contradicts it,
    or is an inference not explicitly stated.

    Returns validation_passed (bool) and a list of ClaimVerdict dicts.
    """
    logger.info("[Validator] Auditing answer for hallucinations...")

    formatted_context = _format_chunks(state["retrieved_chunks"])

    prompt = f"""You are a strict factual auditor. Your job is to check if an answer is fully supported by the provided source passages.

For each factual claim in the answer, check if it is directly supported by the provided context.

A claim FAILS if:
- It is not mentioned in the context at all
- It contradicts something in the context
- It is an inference or speculation not explicitly stated in the context

Return a JSON object in this exact format:
{{
  "all_supported": true/false,
  "verdicts": [
    {{
      "claim": "<the claim text>",
      "verdict": "supported" or "unsupported",
      "reason": "<one sentence explanation>"
    }}
  ]
}}

Only return the JSON. Nothing else.

Source context:
{formatted_context}

Answer to audit:
{state['raw_answer']}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        all_supported = parsed.get("all_supported", False)
        verdicts: list[ClaimVerdict] = parsed.get("verdicts", [])

        if all_supported:
            logger.info("[Validator] PASSED — all claims supported.")
            return {
                "validation_passed": True,
                "claim_verdicts":    verdicts,
                "final_answer":      state["raw_answer"],
                "status":            "success"
            }
        else:
            failed = [v["claim"] for v in verdicts if v["verdict"] == "unsupported"]
            logger.warning(f"[Validator] FAILED — {len(failed)} unsupported claim(s).")
            return {
                "validation_passed": False,
                "claim_verdicts":    verdicts
            }

    except Exception as e:
        logger.error(f"[Validator] Error: {e}. Passing answer through.")
        # On error, pass the answer through rather than looping forever
        return {
            "validation_passed": True,
            "claim_verdicts":    [],
            "final_answer":      state["raw_answer"],
            "status":            "success"
        }
```

---

### `pipeline/agents/regenerator.py` — Agent 7

```python
import logging
from openai import OpenAI
from dotenv import load_dotenv
from pipeline.state import PipelineState, Chunk

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def _format_chunks(chunks: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | Page {chunk['page']} | Type: {chunk['chunk_type']}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def run(state: PipelineState) -> dict:
    """
    Retry generation with the list of unsupported claims explicitly injected into the prompt.
    If max retries are exceeded, return a fallback response pointing to relevant pages.
    """
    retry_count = state.get("retry_count", 0) + 1
    logger.info(f"[Regenerator] Retry attempt {retry_count}/{MAX_RETRIES}")

    # Build the fallback response in case we're already at the limit
    page_refs = sorted(set(c["page"] for c in state["retrieved_chunks"]))
    page_list = ", ".join(f"Page {p}" for p in page_refs)
    fallback_answer = (
        f"I was unable to generate a fully grounded answer to this question. "
        f"The most relevant sections appear to be on {page_list}. "
        f"Please refer to those pages directly."
    )

    if retry_count > MAX_RETRIES:
        logger.warning("[Regenerator] Max retries exceeded. Returning fallback.")
        return {
            "retry_count":      retry_count,
            "final_answer":     fallback_answer,
            "validation_passed": True,  # Force exit from the loop
            "status":           "fallback"
        }

    # Build the list of failed claims for the prompt
    unsupported = [
        v["claim"] for v in state.get("claim_verdicts", [])
        if v["verdict"] == "unsupported"
    ]
    unsupported_list = "\n".join(f"- {claim}" for claim in unsupported)
    formatted_context = _format_chunks(state["retrieved_chunks"])

    prompt = f"""You are a precise document analyst. A previous answer you generated contained unsupported claims. You must now produce a corrected answer.

The following claims were flagged as unsupported or ungrounded:
{unsupported_list}

Rules:
- Do NOT repeat any of the flagged claims.
- Base every statement strictly on the provided context below.
- If the context is insufficient to answer fully, say so explicitly.
- Cite page numbers for each key claim, e.g. (Page 4).

Context:
{formatted_context}

Question: {state['raw_query']}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a precise, grounded document analyst."},
                {"role": "user",   "content": prompt}
            ]
        )
        new_answer = response.choices[0].message.content
        logger.info(f"[Regenerator] New answer: {new_answer[:120]}...")
        return {
            "raw_answer":  new_answer,
            "retry_count": retry_count
        }

    except Exception as e:
        logger.error(f"[Regenerator] Error: {e}. Returning fallback.")
        return {
            "retry_count":       retry_count,
            "final_answer":      fallback_answer,
            "validation_passed": True,
            "status":            "fallback"
        }
```

---

## Deliverable checklist
- [ ] `pipeline/agents/validator.py` — correctly identifies unsupported claims and sets `validation_passed`
- [ ] `pipeline/agents/regenerator.py` — retries with failure context; returns fallback after 3 attempts
- [ ] Test: manually feed the validator a raw_answer that contains a made-up claim and confirm it returns `validation_passed: False`

---

---

# SECTION 08 — LangGraph Graph

## Goal
Wire all seven agents into a LangGraph `StateGraph` with correct conditional routing. This is the orchestration layer — no new agent logic, just connections.

## Dependencies from previous sections
- Sections 01–07 complete — all agents importable

## Context
The graph has two conditional edges:
- After `scope_guard`: route to `END` if out of scope, else to `query_rewriter`
- After `validator`: route to `END` if passed, to `regenerator` if failed and retries remain, or to `END` (fallback) if max retries exceeded

---

## What to build

### `pipeline/graph.py`

```python
import logging
from langgraph.graph import StateGraph, END
from pipeline.state import PipelineState
from pipeline.agents import (
    scope_guard,
    query_rewriter,
    retriever,
    generator,
    validator,
    regenerator,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def _route_after_scope(state: PipelineState) -> str:
    if state["scope_status"] == "out_of_scope":
        logger.info("[Graph] Routing: out_of_scope → END")
        return END
    logger.info("[Graph] Routing: in_scope → query_rewriter")
    return "query_rewriter"


def _route_after_validation(state: PipelineState) -> str:
    if state.get("validation_passed", False):
        logger.info("[Graph] Routing: validation passed → END")
        return END
    if state.get("retry_count", 0) >= MAX_RETRIES:
        logger.info("[Graph] Routing: max retries reached → END")
        return END
    logger.info(f"[Graph] Routing: validation failed (retry {state.get('retry_count', 0)}) → regenerator")
    return "regenerator"


def build_graph():
    """
    Build and compile the LangGraph StateGraph.
    Returns a compiled graph ready to invoke with an initial state dict.
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("scope_guard",    scope_guard.run)
    graph.add_node("query_rewriter", query_rewriter.run)
    graph.add_node("retriever",      retriever.run)
    graph.add_node("generator",      generator.run)
    graph.add_node("validator",      validator.run)
    graph.add_node("regenerator",    regenerator.run)

    # Entry point
    graph.set_entry_point("scope_guard")

    # Edges
    graph.add_conditional_edges("scope_guard", _route_after_scope)
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever",      "generator")
    graph.add_edge("generator",      "validator")
    graph.add_conditional_edges("validator", _route_after_validation)
    graph.add_edge("regenerator",    "validator")

    return graph.compile()
```

---

## Deliverable checklist
- [ ] `pipeline/graph.py` compiles without import errors
- [ ] Smoke test: run the compiled graph with a dummy state dict containing `doc_id`, `topic_summary`, `raw_query`, `retry_count: 0` and confirm it routes correctly
- [ ] Test the out-of-scope path: set `scope_status = "out_of_scope"` manually and confirm it routes to END immediately

---

---

# SECTION 09 — Streamlit UI

## Goal
Build `app.py` — the Streamlit frontend that ties the entire pipeline together. Handles file upload, runs ingestion, accepts queries, invokes the graph, and displays results.

## Dependencies from previous sections
- Sections 01–08 complete — the full pipeline is ready

## Context
Streamlit reruns the entire script on every interaction. Use `st.session_state` to persist the `doc_id`, `topic_summary`, compiled graph, and chat history across reruns. Only re-run ingestion if the uploaded file's MD5 hash is different from the one in session state.

---

## What to build

### `app.py`

```python
import logging
import hashlib
import streamlit as st
from dotenv import load_dotenv

from pipeline.agents.preprocessor import ingest
from pipeline.graph import build_graph
from pipeline.state import PipelineState

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VMARP — Document Q&A",
    page_icon="📄",
    layout="centered"
)

st.title("📄 VMARP — Document Q&A")
st.caption("Upload a PDF, ask questions, get grounded and validated answers.")

# ── Session state defaults ───────────────────────────────────────────────────
if "doc_id"        not in st.session_state: st.session_state.doc_id        = None
if "topic_summary" not in st.session_state: st.session_state.topic_summary = None
if "file_hash"     not in st.session_state: st.session_state.file_hash     = None
if "graph"         not in st.session_state: st.session_state.graph         = build_graph()
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []

# ── Document upload ──────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF (digital or scanned)", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash  = hashlib.md5(file_bytes).hexdigest()

    if file_hash != st.session_state.file_hash:
        # New file uploaded — run ingestion
        with st.spinner(
            "Processing document... "
            "(First run downloads Docling layout models ~1–2 GB. "
            "Subsequent runs are fast.)"
        ):
            result = ingest(file_bytes, uploaded_file.name)

        st.session_state.doc_id        = result["doc_id"]
        st.session_state.topic_summary = result["topic_summary"]
        st.session_state.file_hash     = file_hash
        st.session_state.chat_history  = []  # reset history for new doc

        if result["chunk_count"] > 0:
            st.success(f"✅ Document indexed — {result['chunk_count']} chunks created.")
        else:
            st.info("ℹ️ Document was already indexed. Loaded from cache.")

    # Show document info
    st.markdown(f"**Document loaded:** {uploaded_file.name}")
    st.markdown(f"**Topic:** {st.session_state.topic_summary}")
    st.divider()

# ── Query input ──────────────────────────────────────────────────────────────
if st.session_state.doc_id:
    query = st.text_input(
        "Ask a question about the document",
        placeholder="e.g. What were the key findings in section 3?"
    )

    if st.button("Ask") and query.strip():
        initial_state: PipelineState = {
            "doc_id":            st.session_state.doc_id,
            "topic_summary":     st.session_state.topic_summary,
            "raw_query":         query,
            "rewritten_queries": [],
            "scope_status":      "",
            "scope_reason":      "",
            "retrieved_chunks":  [],
            "raw_answer":        "",
            "claim_verdicts":    [],
            "validation_passed": False,
            "retry_count":       0,
            "final_answer":      "",
            "sources":           [],
            "status":            ""
        }

        with st.spinner("Thinking..."):
            final_state = st.session_state.graph.invoke(initial_state)

        # ── Display answer ───────────────────────────────────────────────────
        status = final_state.get("status", "")

        if status == "out_of_scope":
            st.warning(f"🚫 {final_state['final_answer']}")

        elif status == "fallback":
            st.warning(f"⚠️ {final_state['final_answer']}")

        else:
            st.markdown("### Answer")
            st.markdown(final_state["final_answer"])

            # Sources expander
            with st.expander("📎 Sources used"):
                for src in final_state.get("sources", []):
                    st.markdown(
                        f"- **Page {src['page']}** · `{src['chunk_type']}` — "
                        f"*{src['text_snippet']}*"
                    )

            # Validation details expander
            verdicts = final_state.get("claim_verdicts", [])
            retry_count = final_state.get("retry_count", 0)

            with st.expander("🔍 Validation details"):
                if retry_count == 0:
                    st.markdown("**Status:** ✅ Passed on first attempt")
                else:
                    st.markdown(f"**Status:** ✅ Passed after {retry_count} retry attempt(s)")

                supported   = sum(1 for v in verdicts if v["verdict"] == "supported")
                unsupported = sum(1 for v in verdicts if v["verdict"] == "unsupported")
                st.markdown(f"**Claims checked:** {supported} supported · {unsupported} unsupported")

                for v in verdicts:
                    icon = "✅" if v["verdict"] == "supported" else "❌"
                    st.markdown(f"{icon} **{v['claim']}**  \n*{v['reason']}*")

        # ── Save to chat history ─────────────────────────────────────────────
        st.session_state.chat_history.append({
            "query":  query,
            "answer": final_state.get("final_answer", ""),
            "status": status
        })

    # ── Chat history ─────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.divider()
        st.markdown("### Chat history")
        for i, item in enumerate(reversed(st.session_state.chat_history)):
            status_icon = {"success": "✅", "fallback": "⚠️", "out_of_scope": "🚫"}.get(item["status"], "✅")
            with st.expander(f"{status_icon} Q: {item['query']}"):
                st.markdown(item["answer"])

else:
    st.info("⬆️ Upload a PDF above to get started.")
```

---

## Deliverable checklist
- [ ] `app.py` runs with `streamlit run app.py`
- [ ] Uploading a PDF triggers ingestion and shows topic summary
- [ ] Asking a question returns an answer with sources
- [ ] Validation details expander shows per-claim verdicts
- [ ] Out-of-scope questions show the `🚫` message
- [ ] Chat history persists across questions within a session
- [ ] Uploading a second PDF resets history and re-indexes

---

---

# Full Tech Stack Reference

| Layer | Tool |
|---|---|
| LLM + Validation | `gpt-4o-mini` (OpenAI) |
| Embeddings | `text-embedding-3-small` (OpenAI) |
| Document parsing | `docling` (IBM) — handles digital + scanned, tables, images |
| Orchestration | `langgraph` |
| Vector store | `chromadb` (local, persistent) |
| Keyword search | `rank_bm25` |
| Hybrid merge | Reciprocal Rank Fusion (Python) |
| Document store | `sqlite3` (stdlib) |
| UI | `streamlit` |
| Config | `python-dotenv` |

No system dependencies. No servers. Runs fully locally except for OpenAI API calls.
