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
- If the question references a specific section (e.g., "Section 3.1" or "3.1. Model-based Rewards"), include the section topic as a search query — do NOT generate bare section numbers like "Section 3.1" as they won't match anything.
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
