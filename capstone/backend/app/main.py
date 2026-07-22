"""
FastAPI entrypoint — exposes the LangGraph graph over HTTP via SSE streaming.

Run locally with:
    uvicorn app.main:app --reload --port 8000
"""
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.agents.graph import soccermind_graph

app = FastAPI(title="SoccerMind AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    section: Literal["world_cup", "knowledge"]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest):
    async def event_generator():
        # Tracks whether any node actually streamed LLM tokens during this
        # request. Some paths (router mismatch, World Cup Agent's API-limit
        # fallback) return a ready-made "answer" without ever calling the
        # LLM — those need to be sent as a single event instead, but only if
        # nothing was already streamed for that node (otherwise we'd send
        # the whole answer twice: once token-by-token, once as a duplicate).
        tokens_streamed = False

        async for event in soccermind_graph.astream_events(
            {"question": request.question, "section": request.section},
            version="v2",
        ):
            kind = event["event"]
            node_name = event.get("metadata", {}).get("langgraph_node")

            if kind == "on_chat_model_stream" and node_name in ("world_cup", "knowledge"):
                chunk = event["data"]["chunk"]
                if chunk.content:
                    tokens_streamed = True
                    yield {"event": "token", "data": chunk.content}

            if kind == "on_chain_end" and node_name in ("check_section_match", "world_cup", "knowledge"):
                output = event["data"]["output"]
                if isinstance(output, dict) and output.get("answer") and not tokens_streamed:
                    yield {"event": "token", "data": output["answer"]}

        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())
