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
        section = f" | Section: {chunk['section']}" if chunk.get('section') else ""
        parts.append(
            f"[Chunk {i} | Page {chunk['page']}{section} | Type: {chunk['chunk_type']}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def run(state: PipelineState) -> dict:
    """
    Retry generation with the list of unsupported claims explicitly injected into the prompt.
    If max retries are exceeded, return a fallback response pointing to relevant pages.
    """
    retry_count = state.get("retry_count", 0) + 1
    logger.info(f"[Regenerator] Retry attempt {retry_count}/{MAX_RETRIES}")

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
            "retry_count":       retry_count,
            "final_answer":      fallback_answer,
            "validation_passed": True,
            "status":            "fallback"
        }

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
