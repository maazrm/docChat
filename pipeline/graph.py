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
