"""
Knowledge Agent node — answers questions using RAG over the knowledge base.
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage

from app.config import settings
from app.agents.state import GraphState
from app.tools.rag_search import search_rag

_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)


async def knowledge_agent_node(state: GraphState) -> dict:
    chunks = await search_rag(state["question"], top_k=5)
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)

    system_prompt = (
        "You are a football knowledge assistant. Answer the user's question "
        "using only the context below, and the conversation so far for extra "
        "context (e.g. if the user asks a follow-up like 'what about the "
        "final?'). For tactical questions, focus on explaining WHY something "
        "works, not just describing what it is. Cite the source file in "
        "brackets if you use a specific fact. If the context doesn't answer "
        "the question, say so honestly.\n\n"
        f"Context:\n{context}"
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = await _llm.ainvoke(messages)
    return {"answer": response.content, "messages": [AIMessage(content=response.content)]}