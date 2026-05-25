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
