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
