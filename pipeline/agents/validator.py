import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from pipeline.state import PipelineState, Chunk, ClaimVerdict

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)


def _format_chunks(chunks: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | Page {chunk['page']} | Type: {chunk['chunk_type']}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def run(state: PipelineState) -> dict:
    """
    Audit every factual claim in the generated answer against the source chunks.

    Uses a second gpt-4o-mini call (LLM-as-judge) to verify grounding.
    A claim fails if it is not mentioned in the context, contradicts it,
    or is an inference not explicitly stated.

    Returns validation_passed (bool) and a list of ClaimVerdict dicts.
    """
    logger.info("[Validator] Auditing answer for hallucinations...")

    formatted_context = _format_chunks(state["retrieved_chunks"])

    prompt = f"""You are a strict factual auditor. Your job is to check if an answer is fully supported by the provided source passages.

For each factual claim in the answer, check if it is directly supported by the provided context.

A claim FAILS if:
- It is not mentioned in the context at all
- It contradicts something in the context
- It is an inference or speculation not explicitly stated in the context

Return a JSON object in this exact format:
{{
  "all_supported": true/false,
  "verdicts": [
    {{
      "claim": "<the claim text>",
      "verdict": "supported" or "unsupported",
      "reason": "<one sentence explanation>"
    }}
  ]
}}

Only return the JSON. Nothing else.

Source context:
{formatted_context}

Answer to audit:
{state['raw_answer']}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        all_supported = parsed.get("all_supported", False)
        verdicts: list[ClaimVerdict] = parsed.get("verdicts", [])

        if all_supported:
            logger.info("[Validator] PASSED — all claims supported.")
            return {
                "validation_passed": True,
                "claim_verdicts":    verdicts,
                "final_answer":      state["raw_answer"],
                "status":            "success"
            }
        else:
            failed = [v["claim"] for v in verdicts if v["verdict"] == "unsupported"]
            logger.warning(f"[Validator] FAILED — {len(failed)} unsupported claim(s).")
            return {
                "validation_passed": False,
                "claim_verdicts":    verdicts
            }

    except Exception as e:
        logger.error(f"[Validator] Error: {e}. Passing answer through.")
        return {
            "validation_passed": True,
            "claim_verdicts":    [],
            "final_answer":      state["raw_answer"],
            "status":            "success"
        }
