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
    q_embedding = embed_query(query)
    semantic_ids = query_collection(doc_id, q_embedding, n_results=n)

    bm25_index, chunk_ids = load_bm25(doc_id)
    bm25_ids = bm25_search(query, bm25_index, chunk_ids, n=n)

    merged = reciprocal_rank_fusion([semantic_ids, bm25_ids])
    return merged


def run(state: PipelineState) -> dict:
    """
    Hybrid retrieval over all rewritten queries.
    Runs BM25 + semantic search for each query, merges all results with RRF,
    and returns the top 5 chunks as full Chunk dicts.
    """
    logger.info(f"[Retriever] Queries: {state['rewritten_queries']}")

    try:
        all_ranked_lists = []
        for query in state["rewritten_queries"]:
            ranked = _retrieve_for_query(query, state["doc_id"])
            all_ranked_lists.append(ranked)

        final_ranking = reciprocal_rank_fusion(all_ranked_lists)
        top_ids = final_ranking[:8]

        chunks = get_chunks_by_ids(top_ids)
        logger.info(f"[Retriever] Retrieved {len(chunks)} chunks from pages: {[c['page'] for c in chunks]}")

        return {"retrieved_chunks": chunks}

    except Exception as e:
        logger.error(f"[Retriever] Error: {e}")
        return {"retrieved_chunks": []}
