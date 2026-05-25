import base64
import logging
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
logger = logging.getLogger(__name__)


def caption_figure(image_bytes: bytes) -> str:
    """
    Send an image to gpt-4o-mini vision and return a descriptive caption.
    The caption is stored as a chunk so image content is searchable.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe what this image or figure shows in 1-2 sentences. "
                            "Focus on any data, labels, chart values, or key information visible. "
                            "Be specific and factual."
                        )
                    }
                ]
            }]
        )
        caption = response.choices[0].message.content
        logger.info(f"Captioned figure: {caption[:80]}...")
        return caption
    except Exception as e:
        logger.warning(f"Image captioning failed: {e}")
        return "Figure: content could not be captioned."


def caption_all_figures(document, doc_id: str) -> list[dict]:
    """
    Extract all figures from a Docling document, caption each one,
    and return them as a list of Chunk dicts ready for indexing.
    """
    caption_chunks = []
    for i, picture in enumerate(document.pictures):
        try:
            pil_image = picture.get_image(document)
            buf = BytesIO()
            pil_image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            caption = caption_figure(img_bytes)
            page_no = picture.prov[0].page_no if picture.prov else 0
            caption_chunks.append({
                "id":         f"{doc_id}_figure_{i}",
                "text":       caption,
                "page":       page_no,
                "chunk_type": "image_caption",
                "section":    None,
                "doc_id":     doc_id
            })
        except Exception as e:
            logger.warning(f"Skipping figure {i}: {e}")

    logger.info(f"Captioned {len(caption_chunks)} figures")
    return caption_chunks
