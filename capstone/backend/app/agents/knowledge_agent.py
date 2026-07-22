"""
Knowledge Agent node — answers questions using RAG over the knowledge base.
"""
from langchain_openai import ChatOpenAI

from app.config import settings
from app.agents.state import GraphState
from app.tools.rag_search import search_rag

_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)


async def knowledge_agent_node(state: GraphState) -> dict:
    """Retrieve relevant chunks, then generate an answer grounded only in them — the R and G of RAG."""
    chunks = await search_rag(state["question"], top_k=5)

    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)

    prompt = (
        "You are a football knowledge assistant. Answer the user's question "
        "using only the context below. Cite the source file in brackets if "
        "you use a specific fact. If the context doesn't answer the question, "
        "say so honestly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}"
    )

    response = await _llm.ainvoke(prompt)
    return {"answer": response.content}
