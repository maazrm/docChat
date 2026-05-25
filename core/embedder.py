from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using text-embedding-3-small.
    Processes in batches of 100 to stay within API limits.
    """
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        all_embeddings.extend([r.embedding for r in response.data])
    return all_embeddings

def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
