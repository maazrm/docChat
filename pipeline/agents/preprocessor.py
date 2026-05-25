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
