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
