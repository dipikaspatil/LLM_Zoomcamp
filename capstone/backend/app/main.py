"""
FastAPI entrypoint — exposes the LangGraph graph over HTTP via SSE streaming.

Run locally with:
    uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.config import settings
from app.agents.graph import build_graph
from app.tools.conversations import ensure_conversations_table, touch_conversation

VALID_NODES = ("world_cup", "knowledge", "club_football")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph(checkpointer)

        async with AsyncConnectionPool(settings.DATABASE_URL, open=False) as db_pool:
            await db_pool.open()
            await ensure_conversations_table(db_pool)
            app.state.db_pool = db_pool
            yield


app = FastAPI(title="SoccerMind AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    thread_id: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/conversations")
async def list_conversations(request: Request):
    """Returns all conversations, most recently active first."""
    db_pool = request.app.state.db_pool
    async with db_pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT thread_id, title, created_at, updated_at "
            "FROM conversations ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
    return [
        {
            "thread_id": row[0],
            "title": row[1],
            "created_at": row[2].isoformat(),
            "updated_at": row[3].isoformat(),
        }
        for row in rows
    ]


@app.get("/conversations/{thread_id}/messages")
async def get_conversation_messages(request: Request, thread_id: str):
    """
    Returns a conversation's full message history, read directly from the
    LangGraph checkpointer (not our own table, which only holds title/timestamps).
    """
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)

    messages = state.values.get("messages", [])
    return [
        {"role": "user" if msg.type == "human" else "assistant", "text": msg.content}
        for msg in messages
    ]


@app.post("/chat")
async def chat(request: Request, chat_request: ChatRequest):
    graph = request.app.state.graph
    db_pool = request.app.state.db_pool

    await touch_conversation(db_pool, chat_request.thread_id, chat_request.question)

    async def event_generator():
        tokens_streamed = False

        async for event in graph.astream_events(
            {
                "question": chat_request.question,
                "messages": [HumanMessage(content=chat_request.question)],
            },
            config={"configurable": {"thread_id": chat_request.thread_id}},
            version="v2",
        ):
            kind = event["event"]
            node_name = event.get("metadata", {}).get("langgraph_node")

            if kind == "on_chat_model_stream" and node_name in VALID_NODES:
                chunk = event["data"]["chunk"]
                if chunk.content:
                    tokens_streamed = True
                    yield {"event": "token", "data": chunk.content}

            if kind == "on_chain_end" and node_name in ("classify_section",) + VALID_NODES:
                output = event["data"]["output"]
                if isinstance(output, dict) and output.get("answer") and not tokens_streamed:
                    yield {"event": "token", "data": output["answer"]}

        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())