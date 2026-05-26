import logging
from openai import OpenAI
from dotenv import load_dotenv
from pipeline.state import PipelineState, Chunk

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)


def _format_chunks(chunks: list[Chunk]) -> str:
    """Format retrieved chunks into a readable context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | Page {chunk['page']} | Type: {chunk['chunk_type']}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def run(state: PipelineState) -> dict:
    """
    Generate a grounded answer from the retrieved chunks.
    Every claim must cite a page number. No speculation beyond the context.
    """
    logger.info(f"[Generator] Generating answer for: {state['raw_query']}")

    formatted_context = _format_chunks(state["retrieved_chunks"])

    prompt = f"""You are a precise document analyst. Answer the user's question using ONLY the provided context passages.

Rules:
- Base every claim strictly on the provided context.
- If the context does not contain enough information to fully answer, say so explicitly.
- Do NOT speculate or add information from outside the context.
- For each key claim, cite the page number in parentheses, e.g. (Page 4).
- Keep the answer clear and well-structured.

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
        answer = response.choices[0].message.content
        logger.info(f"[Generator] Answer: {answer[:120]}...")

        sources = [
            {
                "page":         c["page"],
                "chunk_type":   c["chunk_type"],
                "text_snippet": c["text"][:150] + "..."
            }
            for c in state["retrieved_chunks"]
        ]

        return {"raw_answer": answer, "sources": sources}

    except Exception as e:
        logger.error(f"[Generator] Error: {e}")
        return {
            "raw_answer": "An error occurred while generating the answer. Please try again.",
            "sources":    []
        }
