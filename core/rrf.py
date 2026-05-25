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
