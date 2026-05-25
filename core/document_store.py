import sqlite3
import json
from pathlib import Path

DB_PATH = Path("data/documents.db")

def _connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(str(DB_PATH))

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
