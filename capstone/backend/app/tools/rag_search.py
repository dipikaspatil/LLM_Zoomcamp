"""
search_rag(): the "R" (Retrieval) in RAG.

Given a user's question, embed it the same way the knowledge base chunks were
embedded, then ask Qdrant for the most similar chunks. This is called by the
Knowledge Agent at request time — unlike ingestion, this must be async since
it runs inside the live FastAPI/LangGraph request path, not a one-off script.
"""
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.tools.vector_store import COLLECTION_NAME, EMBEDDING_MODEL

# Created once at import time and reused across requests — avoids reconnecting
# to Qdrant or re-initializing the embedder on every single question
_embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
_client = AsyncQdrantClient(url=settings.QDRANT_URL)


async def search_rag(query: str, section: str | None = None, top_k: int = 5) -> list[dict]:
    """
    Return the top_k most relevant knowledge base chunks for a question.

    section: optional filter (e.g. "tactical_concepts") so results only come
    from the section the user picked in the UI, instead of the whole knowledge base.
    """
    query_vector = _embedder.embed_query(query)  # turn the question itself into a vector

    query_filter = None
    if section:
        query_filter = Filter(
            must=[FieldCondition(key="section", match=MatchValue(value=section))]
        )

    # query_points() replaces the older, now-removed search() method —
    # note it returns a response object with a .points list, not a plain list
    response = await _client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )

    return [
        {
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "section": hit.payload["section"],
            "score": hit.score,
        }
        for hit in response.points
    ]
