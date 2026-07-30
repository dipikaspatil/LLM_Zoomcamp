"""
Lightweight conversation metadata (titles, timestamps) — separate from
LangGraph's own checkpoint tables, which store raw graph state, not
human-readable metadata. Uses a small dedicated table we manage ourselves.
"""
from psycopg_pool import AsyncConnectionPool

MAX_TITLE_LENGTH = 50


async def ensure_conversations_table(pool: AsyncConnectionPool) -> None:
    """Creates the conversations table if it doesn't already exist. Called once at startup."""
    async with pool.connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )


def _make_title(first_message: str) -> str:
    """Derives a short title from the first user message — truncate, no LLM call needed."""
    title = first_message.strip()
    if len(title) > MAX_TITLE_LENGTH:
        title = title[:MAX_TITLE_LENGTH].rstrip() + "..."
    return title


async def touch_conversation(pool: AsyncConnectionPool, thread_id: str, question: str) -> None:
    """
    Records that this thread_id was just used. If it's a brand-new
    conversation, sets its title from this (first) question. If it already
    exists, just bumps updated_at so it sorts to the top of a "recent
    conversations" list — the title is never overwritten after creation.
    """
    title = _make_title(question)
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO conversations (thread_id, title)
            VALUES (%s, %s)
            ON CONFLICT (thread_id) DO UPDATE SET updated_at = now()
            """,
            (thread_id, title),
        )
