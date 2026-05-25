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
    Classify whether the user's query is answerable from the uploaded document.
    Returns scope_status ("in_scope" | "out_of_scope") and scope_reason.
    Short-circuits the pipeline if out of scope.
    """
    logger.info(f"[ScopeGuard] Query: {state['raw_query']}")

    prompt = f"""You are a strict scope classifier. Your job is to decide if a user's question can be answered using the provided document.

Document summary: {state['topic_summary']}

User question: {state['raw_query']}

Can this question be answered from the document described above?
Reply with a JSON object: {{"in_scope": true/false, "reason": "<one sentence>"}}
Only reply with the JSON. Nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        in_scope = parsed.get("in_scope", False)
        reason = parsed.get("reason", "")

        if in_scope:
            logger.info(f"[ScopeGuard] IN SCOPE — {reason}")
            return {
                "scope_status": "in_scope",
                "scope_reason": reason
            }
        else:
            logger.info(f"[ScopeGuard] OUT OF SCOPE — {reason}")
            return {
                "scope_status":  "out_of_scope",
                "scope_reason":  reason,
                "final_answer":  (
                    f"This question is outside the scope of the uploaded document. "
                    f"The document covers: {state['topic_summary']}. "
                    f"Please ask something related to its content."
                ),
                "status": "out_of_scope"
            }

    except Exception as e:
        logger.error(f"[ScopeGuard] Error: {e}")
        return {
            "scope_status": "in_scope",  # fail open — let the pipeline continue
            "scope_reason": "Scope check failed; proceeding."
        }
