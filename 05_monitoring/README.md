# Module 5: Monitoring

This module covers **online evaluation** — monitoring a RAG system once real
users are hitting it. Where module 4 measured quality offline (Hit Rate, MRR,
LLM-as-judge on a fixed dataset), this module wraps a live RAG pipeline in a
UI, records every call to a database, and visualizes it on a dashboard.

By the end you'll have:

- A Streamlit chat app in front of the RAG pipeline from earlier modules
- Every call instrumented for latency, tokens, and cost
- Conversations + feedback persisted in PostgreSQL
- Two dashboards: a quick one in Streamlit, a richer one in Grafana
- Two feedback signals: user thumbs up/down and an LLM-as-judge relevance score
- A Docker Compose setup to run the whole stack with one command

> We only build the RAG version here. Monitoring an **agent** works the same
> way — capture each tool call like we capture each LLM call — and is left as
> homework (see [Lesson 14](#lesson-14--next-steps)).

---

## Table of contents

| # | Lesson | Video | Adds |
|---|--------|:-----:|------|
| 1 | [Intro](#lesson-1--intro) | ✅ | Why monitoring, what we're building |
| 2 | [Assistant Setup](#lesson-2--assistant-setup) | ✅ | `assistant.py`, reuses `RAGBase` |
| 3 | [Chat App](#lesson-3--chat-app) | ✅ | Streamlit UI — `app.py` |
| 4 | [Capturing Metrics](#lesson-4--capturing-metrics) | ✅ | `metrics.py` — latency/tokens/cost |
| 5 | [Storing Data in PostgreSQL](#lesson-5--storing-data-in-postgresql) | ✅ | Docker Postgres, `db_init.py`, `db_save.py` |
| 6 | [Querying Data](#lesson-6--querying-data) | ✅ | `db_query.py` |
| 7 | [Streamlit Dashboard](#lesson-7--streamlit-dashboard) | ✅ | `dashboard.py` |
| 8 | [User Feedback](#lesson-8--user-feedback) | ✅ | thumbs up/down, `feedback` table |
| 9 | [Built-in Judge](#lesson-9--built-in-judge) | ✅ | `judge.py`, online LLM-as-judge |
| 10 | [Feedback Dashboard](#lesson-10--feedback-dashboard) | — | feedback panels in Streamlit |
| 11 | [Synthetic Data](#lesson-11--synthetic-data-generation) | — | `generate_data.py` |
| 12 | [Grafana](#lesson-12--grafana-dashboards) | ✅ | Grafana + Postgres data source, panels |
| 13 | [Docker Compose](#lesson-13--docker-compose) | — | one-command stack |
| 14 | [Next Steps](#lesson-14--next-steps) | ✅ | frameworks, production concerns, homework |

Source: [DataTalksClub/llm-zoomcamp/05-monitoring/lessons](https://github.com/DataTalksClub/llm-zoomcamp/tree/main/05-monitoring/lessons)

---

## Prerequisites

- Completed modules 1–4 (in particular `ingest.py`, `rag_helper.py` /
  `RAGBase` from `01-agentic-rag/code/`, and `evaluation_utils.py` from
  `04-evaluation/code/`)
- `uv` for dependency management
- Docker (for Postgres and Grafana)
- An `OPENAI_API_KEY` in a `.env` file

---

## Lesson 1 — Intro

Video: [Watch](https://www.youtube.com/watch?v=lbEj3Waxs1U&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

Offline evaluation (module 4) tells you how good the system *should* be.
Monitoring tells you how it's *actually* doing with real traffic — latency,
cost, and whether users find answers useful.

For every question, we want to capture:

- Instructions, prompt, and model used
- Input/output tokens and cost
- Response time
- User feedback (thumbs up/down)
- Relevance (does the answer address the question?)

This needs three new pieces on top of the existing RAG pipeline: a **UI**, a
**database**, and a **dashboard**.

---

## Lesson 2 — Assistant Setup

Video: [Watch](https://www.youtube.com/watch?v=jMO8rqPmR-4&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

Reuses the RAG pipeline (search → prompt → LLM) built in earlier modules via
`ingest.py` and `rag_helper.py` (`RAGBase`).

Download the helpers if you don't have them:

```bash
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main

wget ${PREFIX}/01-agentic-rag/code/ingest.py
wget ${PREFIX}/01-agentic-rag/code/rag_helper.py
```

```bash
uv add python-dotenv
```

`assistant.py`:

```python
import sys

from dotenv import load_dotenv
from openai import OpenAI

from ingest import load_faq_data, build_index
from rag_helper import RAGBase

def create_assistant():
    load_dotenv()

    documents = load_faq_data()
    index = build_index(documents)

    return RAGBase(
        index=index,
        llm_client=OpenAI(),
    )

if __name__ == "__main__":
    assistant = create_assistant()

    query = "How do I join the course?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    answer = assistant.rag(query)
    print(answer)
```

`RAGBase` already ships a system prompt for course Q&A — don't pass your own
instructions, it'd be redundant.

**Makefile** (start it here, we keep appending targets in later lessons):

```makefile
run:
	uv run python assistant.py
```

```bash
make run
# or
uv run python assistant.py "How do I join the course?"
```

---

## Lesson 3 — Chat App

Video: [Watch](https://www.youtube.com/watch?v=JCB4JZlMsIQ&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

Wraps the CLI in a minimal Streamlit UI — intentionally plain, hand it to a
coding assistant later if you want a nicer layout.

```bash
uv add streamlit
```

`app.py`:

```python
import streamlit as st
from assistant import create_assistant

assistant = create_assistant()

st.title("Course Assistant")

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    with st.spinner("Processing..."):
        answer = assistant.rag(user_input)
        st.success("Completed!")
        st.write(answer)
```

**Makefile** addition:

```makefile
chat:
	uv run streamlit run app.py
```

```bash
make chat
```

At this point the RAG works but nothing about the call is tracked — no
latency, tokens, or cost. That's what monitoring adds next.

---

## Lesson 4 — Capturing Metrics

Video: [Watch](https://www.youtube.com/watch?v=JGh6-DqaueA&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

`metrics.py` — a dataclass to hold everything captured per call, plus a
`RAGBase` subclass that overrides only the LLM-calling method.

```python
import time
from dataclasses import dataclass, field
from datetime import datetime

from rag_helper import RAGBase

@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)
```

Cost calculation — provider charges per-million-token rates:

```python
def calculate_cost(model, usage):
    cost = 0
    if "gpt-5.4-mini" in model:
        cost = (usage.input_tokens * 0.15 + usage.output_tokens * 0.60) / 1_000_000
    return cost
```

> Duplicated across lessons rather than centralized, to keep each lesson
> self-contained — pull it into one shared module in a real project.

Subclass that captures metrics on every `rag()` call:

```python
class RAGWithMetrics(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_call: LLMCallRecord = None

    def llm(self, prompt):
        start_time = time.time()
        response = self._call_llm(prompt)
        response_time = time.time() - start_time
        self._log_response(prompt, response, response_time)
        return response.output_text

    def _call_llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        return response

    def _log_response(self, prompt, response, response_time):
        usage = response.usage
        cost = calculate_cost(self.model, usage)

        call_record = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=response.output_text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            response_time=response_time,
            cost=cost,
        )

        print(call_record)
        self.last_call = call_record
```

> `last_call` is stashed on `self` rather than returned, to avoid changing
> `rag()`'s return type. Fine for a single-user Streamlit demo; not
> thread-safe for concurrent callers.

Update `assistant.py` to use `RAGWithMetrics` instead of `RAGBase`:

```python
from metrics import RAGWithMetrics
# ...
return RAGWithMetrics(index=index, llm_client=OpenAI())
```

Display metrics in `app.py`, after the answer:

```python
record = assistant.last_call
st.write(f"Response time: {record.response_time:.2f}s")
st.write(f"Prompt tokens: {record.prompt_tokens}")
st.write(f"Completion tokens: {record.completion_tokens}")
st.write(f"Cost: ${record.cost:.4f}")
```

Metrics still vanish when the app closes — next we persist them.

---

## Lesson 5 — Storing Data in PostgreSQL

Video: [Watch](https://www.youtube.com/watch?v=iXRu_AbMtuU&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

A dedicated Postgres instance (only for monitoring — nothing else touches
it), because it handles structured data well and Grafana connects to it
easily.

```bash
docker network create monitoring

docker run -it \
    --name course-assistant-pg \
    --network monitoring \
    -e POSTGRES_USER=user \
    -e POSTGRES_PASSWORD=password \
    -e POSTGRES_DB=course_assistant \
    -p 5432:5432 \
    -v pgdata:/var/lib/postgresql/data \
    postgres:17
```

**Makefile** additions:

```makefile
network:
	docker network create monitoring

postgres: network
	docker run -it \
		--name course-assistant-pg \
		--network monitoring \
		-e POSTGRES_USER=user \
		-e POSTGRES_PASSWORD=password \
		-e POSTGRES_DB=course_assistant \
		-p 5432:5432 \
		-v pgdata:/var/lib/postgresql/data \
		postgres:17
```

```bash
make postgres
uv add "psycopg[binary]"
```

### Schema

`conversations` holds everything from `LLMCallRecord` plus the raw question
and course. Two design notes: `course` exists so one assistant can serve
multiple courses later; `timestamp` is `TIMESTAMP WITH TIME ZONE` — without
tz-awareness Grafana won't line points up on its time axis.

```sql
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    course TEXT NOT NULL,
    model TEXT NOT NULL,
    instructions TEXT NOT NULL,
    prompt TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    response_time FLOAT NOT NULL,
    cost FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
)
```

`db_init.py`:

```python
import os
import psycopg
from datetime import datetime

DB_TIMEZONE = datetime.now().astimezone().tzinfo
print(f"Using timezone: {DB_TIMEZONE}")

def get_db_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        dbname=os.getenv("POSTGRES_DB", "course_assistant"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )

def init_db(drop=False):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if drop:
                cur.execute("DROP TABLE IF EXISTS conversations")

            cur.execute("""
                CREATE TABLE conversations (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    course TEXT NOT NULL,
                    model TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    response_time FLOAT NOT NULL,
                    cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized")
```

```bash
uv run python db_init.py
```

> Run once — the `pgdata` volume persists across container restarts. Only
> re-run when the schema changes (⚠️ `drop=True` wipes existing data).

`db_save.py` — insert a record, `RETURNING id` since feedback later needs to
reference the conversation:

```python
from datetime import datetime
from db_init import get_db_connection, DB_TIMEZONE

def save_conversation(record, question, course):
    timestamp = datetime.now(DB_TIMEZONE)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (
                    question, answer, course, model, instructions, prompt,
                    prompt_tokens, completion_tokens, total_tokens,
                    response_time, cost, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id
                """,
                (
                    question,
                    record.answer,
                    course,
                    record.model,
                    record.instructions,
                    record.prompt,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.response_time,
                    record.cost,
                    timestamp,
                ),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return conversation_id
```

> `question` is passed separately from `record.prompt` — `prompt` is the
> full text sent to the model; `question` is what the user actually typed.

Wire into `assistant.py`'s `__main__` and into `app.py` after `assistant.rag()`:

```python
from db_save import save_conversation
# CLI:
save_conversation(assistant.last_call, query, "llm-zoomcamp")
# Streamlit:
conversation_id = save_conversation(record, user_input, "llm-zoomcamp")
st.session_state.conversation_id = conversation_id
```

Verify:

```bash
docker exec -it course-assistant-pg psql -U user -d course_assistant \
    -c "SELECT id, question, response_time, cost FROM conversations;"
```

---

## Lesson 6 — Querying Data

Video: [Watch](https://www.youtube.com/watch?v=18vEtjPJwLc&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

`db_query.py` reads rows back and converts each tuple into an `LLMCallRecord`
so callers don't have to remember column indices.

```python
from dataclasses import dataclass

from db_init import get_db_connection
from metrics import LLMCallRecord

def row_to_record(row):
    return LLMCallRecord(
        model=row[4],
        prompt=row[6],
        instructions=row[5],
        answer=row[2],
        prompt_tokens=row[7],
        completion_tokens=row[8],
        total_tokens=row[9],
        response_time=row[10],
        cost=row[11],
        timestamp=row[12],
    )

def get_conversations(limit=10):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, answer, course, model,
                       instructions, prompt,
                       prompt_tokens, completion_tokens, total_tokens,
                       response_time, cost, timestamp
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [row_to_record(row) for row in rows]

if __name__ == "__main__":
    records = get_conversations()
    for record in records:
        print(record)
```

```bash
uv run python db_query.py
```

> No index on `timestamp` (only on `id`). Fine at small scale; since `id`
> increases monotonically, ordering by `id` would be faster at volume, or
> add an index on `timestamp`.

---

## Lesson 7 — Streamlit Dashboard

Video: [Watch](https://www.youtube.com/watch?v=OrWlgDKZclI&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

A quick dashboard before reaching for Grafana — often all you need for a
smaller project. (If you stop here, you don't even need Postgres — SQLite +
this dashboard is a fine lightweight setup; Postgres is only needed because
Grafana connects to it more easily later.)

Aggregate query, added to `db_query.py`:

```python
@dataclass
class Stats:
    total: int
    avg_response_time: float
    total_cost: float
    avg_tokens: float

def get_stats():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*),
                    AVG(response_time),
                    SUM(cost),
                    AVG(total_tokens)
                FROM conversations
            """)
            row = cur.fetchone()
    finally:
        conn.close()

    return Stats(
        total=row[0],
        avg_response_time=row[1],
        total_cost=row[2],
        avg_tokens=row[3],
    )
```

`dashboard.py`:

```python
import streamlit as st
from dataclasses import asdict
import pandas as pd
from db_query import get_conversations, get_stats

st.title("Course Assistant Dashboard")

stats = get_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total conversations", stats.total)
col2.metric("Avg response time", f"{stats.avg_response_time:.2f}s")
col3.metric("Total cost", f"${stats.total_cost:.4f}")
col4.metric("Avg tokens", f"{stats.avg_tokens:.0f}")

records = get_conversations(limit=100)
df = pd.DataFrame([asdict(r) for r in records])

st.subheader("Cost over time")
st.line_chart(df, x="timestamp", y="cost")

st.subheader("Response time over time")
st.line_chart(df, x="timestamp", y="response_time")

st.subheader("Recent conversations")
records = get_conversations(limit=20)

for record in records:
    st.write(f"**{record.prompt[:80]}...**")
    st.write(f"{record.answer[:200]}...")
    st.write(f"Time: {record.response_time:.2f}s | Cost: ${record.cost:.4f}")
    st.divider()
```

```bash
# port 8501 is already used by the chat app
uv run streamlit run dashboard.py --server.port 8502
```

---

## Lesson 8 — User Feedback

Video: [Watch](https://www.youtube.com/watch?v=GEifsHDadBw&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

None of the metrics so far say whether an answer was *good*. Add thumbs
up/down, recorded in a shared `feedback` table (a `source` column
distinguishes `"user"` from `"judge"`, added next lesson).

> Noisy signal (accidental clicks, misrated answers) but still valuable —
> especially a spike of thumbs-down in the last hour as an alert to go
> check what broke, and as raw material for aligning an LLM judge later.

Add to `db_init.py`:

```python
def init_feedback():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS feedback")

            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id),
                    source TEXT NOT NULL,
                    relevance TEXT,
                    explanation TEXT,
                    score INTEGER,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    init_feedback()
    print("Database initialized")
```

```bash
uv run python db_init.py
```

- `source`: `"user"` (human) or `"judge"` (LLM, lesson 9)
- `score`: `+1` thumbs up, `-1` thumbs down
- `relevance` / `explanation`: used by the judge

`db_feedback.py`:

```python
from datetime import datetime
from db_init import get_db_connection, DB_TIMEZONE

def save_feedback(conversation_id, source, relevance=None,
                  explanation=None, score=None):
    timestamp = datetime.now(DB_TIMEZONE)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    conversation_id, source, relevance,
                    explanation, score, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s
                )
                """,
                (conversation_id, source, relevance,
                 explanation, score, timestamp),
            )
        conn.commit()
    finally:
        conn.close()
```

In `app.py`, keep `conversation_id` in `st.session_state` (Streamlit reruns
the whole script on every click, so this is how the id survives from the
answer to the button press) and add the buttons:

```python
from db_feedback import save_feedback

# after save_conversation(...):
# st.session_state.conversation_id = conversation_id

col1, col2 = st.columns(2)
with col1:
    if st.button("+1"):
        cid = st.session_state.conversation_id
        save_feedback(cid, "user", score=1)
        st.write("Thanks!")

with col2:
    if st.button("-1"):
        cid = st.session_state.conversation_id
        save_feedback(cid, "user", score=-1)
        st.write("Thanks for the feedback!")
```

---

## Lesson 9 — Built-in Judge

Video: [Watch](https://www.youtube.com/watch?v=YLOLQyrMDuY&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

Same idea as offline LLM-as-judge (module 4), run online after every answer.
Key difference: no ground truth available online — the judge only sees
question + answer, so the instructions need to be more explicit about what
"good" means.

```bash
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main
wget ${PREFIX}/04-evaluation/code/evaluation_utils.py
```

`judge.py`:

```python
import json

from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from dotenv import load_dotenv

from evaluation_utils import llm_structured_retry

class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str

judge_instructions = """
You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: the answer addresses the question
- PARTLY_RELEVANT: the answer partially addresses the question
- NON_RELEVANT: the answer does not address the question
""".strip()

judge_prompt = """
Question: {question}
Generated Answer: {answer}
""".strip()

def evaluate_relevance(question, answer, client=None):
    if client is None:
        client = OpenAI()

    prompt = judge_prompt.format(
        question=question,
        answer=answer
    )

    result, usage = llm_structured_retry(
        client,
        judge_instructions,
        prompt,
        RelevanceVerdict,
    )

    return result.relevance, result.explanation

if __name__ == "__main__":
    load_dotenv()

    question = "Can I still join the course?"
    answer = "Yes, you can still join. The course is self-paced."

    relevance, explanation = evaluate_relevance(question, answer)
    print(relevance)
    print(explanation)
```

```bash
uv run python judge.py
```

> `explanation` matters even when unused downstream — forcing the model to
> reason before committing to a label tends to improve the label. And this
> judge is deliberately basic: budget time to align it against real user
> labels before trusting it.

Integrate into `app.py` (writes to the same `feedback` table with
`source="judge"`):

```python
from judge import evaluate_relevance
from db_feedback import save_feedback

# after save_conversation / st.session_state.conversation_id = ...:
relevance, explanation = evaluate_relevance(user_input, answer)
save_feedback(conversation_id, "judge",
                relevance=relevance, explanation=explanation)
st.write(f"Relevance: {relevance}")
st.write(f"Explanation: {explanation}")
```

**Production notes:**
- The judge is an extra LLM call per question → run it asynchronously
  (answer first, score in background) in a real system.
- Track the judge's own cost separately (e.g. a `judge_feedback` table).
- You don't have to judge every answer — sampling (e.g. 1 in 10) keeps most
  of the signal for a fraction of the cost.

---

## Lesson 10 — Feedback Dashboard

_No video._

Surface both feedback signals (user + judge) in the Streamlit dashboard
alongside cost/latency.

Add to `db_query.py`:

```python
def get_relevance_stats():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT relevance, COUNT(*)
                FROM feedback
                WHERE source = 'judge'
                GROUP BY relevance
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    return dict(rows)

def get_user_feedback_stats():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END)
                FROM feedback
                WHERE source = 'user'
            """)
            row = cur.fetchone()
    finally:
        conn.close()
    return row
```

Add to `dashboard.py`:

```python
from db_query import get_conversations, get_stats, get_relevance_stats, get_user_feedback_stats

st.subheader("Judge relevance")
relevance = get_relevance_stats()
st.bar_chart(relevance)

st.subheader("User feedback")
thumbs_up, thumbs_down = get_user_feedback_stats()
col1, col2 = st.columns(2)
col1.metric("Thumbs up", int(thumbs_up or 0))
col2.metric("Thumbs down", int(thumbs_down or 0))
```

---

## Lesson 11 — Synthetic Data Generation

_No video._

Fill the database with fake traffic so the dashboard has something to show,
and so Grafana (next lesson) has near-real-time data to visualize.

`generate_data.py`:

```python
import time
import random

from metrics import LLMCallRecord
from db_save import save_conversation
from db_feedback import save_feedback

SAMPLE_QUESTIONS = [
    "How do I install Docker?",
    "Can I still join the course?",
    "What are the prerequisites?",
    "How do I submit homework?",
    "When are the office hours?",
]

SAMPLE_ANSWERS = [
    "You can install Docker by downloading Docker Desktop from the official website.",
    "Yes, you can join at any time. The materials remain available.",
    "You need basic Python knowledge and familiarity with the command line.",
    "Submit your homework through the course portal before the deadline.",
    "Office hours are held weekly. Check the calendar for details.",
]

RELEVANCE = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]

def fake_record(question, answer):
    return LLMCallRecord(
        model="gpt-5.4-mini",
        prompt=question,
        instructions="",
        answer=answer,
        prompt_tokens=random.randint(50, 200),
        completion_tokens=random.randint(50, 300),
        total_tokens=random.randint(100, 500),
        response_time=random.uniform(0.5, 5.0),
        cost=random.uniform(0.0001, 0.01),
    )

def random_score():
    # weighted toward positive, simulating mostly-happy users
    return random.choice([1, 1, 1, 1, -1])

def generate_one():
    question = random.choice(SAMPLE_QUESTIONS)
    answer = random.choice(SAMPLE_ANSWERS)
    record = fake_record(question, answer)

    conversation_id = save_conversation(
        record, question, "llm-zoomcamp"
    )

    if random.random() < 0.7:
        relevance = random.choice(RELEVANCE)
        save_feedback(
            conversation_id, "judge",
            relevance=relevance,
            explanation=f"Answer is {relevance.lower()}.",
        )

    if random.random() < 0.5:
        score = random_score()
        save_feedback(conversation_id, "user", score=score)

def generate_live():
    print("Starting live data generation (Ctrl+C to stop)...", flush=True)
    while True:
        generate_one()
        time.sleep(1)

if __name__ == "__main__":
    try:
        generate_live()
    except KeyboardInterrupt:
        print("Stopped.")
```

```bash
uv run python generate_data.py
```

Leave it running while you build Grafana panels in the next lesson — the
charts update live.

---

## Lesson 12 — Grafana Dashboards

Video: [Watch](https://www.youtube.com/watch?v=Pmh2jT8tEiw&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

Grafana over the Streamlit dashboard when you want more panel types, more
data sources, and alerting. Heavier to run — use it once the simple
dashboard stops being enough.

```bash
docker run -d \
    --name grafana \
    --network monitoring \
    -p 3000:3000 \
    -v grafana_data:/var/lib/grafana \
    grafana/grafana
```

Access at `http://localhost:3000` (first login `admin` / `admin`, then set a
new password — `admin`/`admin` again is fine locally).

### Data source

Configuration → Data Sources → Add data source → PostgreSQL:

| Field | Value |
|---|---|
| Host | `course-assistant-pg:5432` |
| Database | `course_assistant` |
| User | `user` |
| Password | `password` |
| SSL Mode | disable |

Save & Test → "Database Connection OK".

### Panel-building habits

- Alias the time column as `time` — Grafana uses it for the x-axis.
- Filter on the selected range with `$__timeFrom()` / `$__timeTo()` so
  panels follow the picker at the top.
- `$__timeGroup(column, interval)` buckets rows by time interval.

### Panels

**Response Time** (Time series) — one row per LLM call, plot raw:

```sql
SELECT
  timestamp AS time,
  response_time
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
ORDER BY timestamp
```

**Token Usage** (Time series) — bucketed average, avoids overplotting on
long ranges:

```sql
SELECT
  $__timeGroup(timestamp, $__interval) AS time,
  AVG(total_tokens) AS avg_tokens
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY 1
ORDER BY 1
```

**Cost** (Time series) — bucketed sum:

```sql
SELECT
  $__timeGroup(timestamp, $__interval) AS time,
  SUM(cost) AS total_cost
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
  AND cost > 0
GROUP BY 1
ORDER BY 1
```

**Model Usage** (Bar chart):

```sql
SELECT
  model,
  COUNT(*) as count
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY model
```

**Relevance Distribution** (Pie chart):

```sql
SELECT
  relevance,
  COUNT(*) as count
FROM feedback
WHERE source = 'judge'
  AND timestamp BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY relevance
```

**User Feedback** (Gauge or Pie chart):

```sql
SELECT
  SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END) as thumbs_up,
  SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END) as thumbs_down
FROM feedback
WHERE source = 'user'
  AND timestamp BETWEEN $__timeFrom() AND $__timeTo()
```

**Recent Conversations** (Table):

```sql
SELECT
  timestamp AS time,
  question,
  answer,
  response_time,
  cost
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
ORDER BY timestamp DESC
LIMIT 5
```

### Dashboard settings

- Auto-refresh every 30s; default time range "Last 6 hours"
- Suggested layout: recent conversations table (wide, top) → model usage +
  relevance pie (middle) → response time / token usage / cost (bottom)
- Panel types aren't fixed — try bar/pie/time-series on the same query and
  keep whatever reads best

---

## Lesson 13 — Docker Compose

_No video._

Running Postgres, Grafana, and Streamlit by hand gets old (remembering the
network, retyping commands, cleaning up name conflicts). Compose starts all
three together.

### Project structure

```text
code/
├── docker-compose.yaml
├── Dockerfile
├── .env
├── pyproject.toml
├── uv.lock
├── .python-version
├── app.py           # Streamlit app
├── assistant.py     # RAG pipeline + LLM
├── db_init.py       # Database init
├── db_save.py       # Save conversations
└── dashboard.py     # Streamlit dashboard
```

### Dockerfile

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### `.env`

```text
POSTGRES_DB=course_assistant
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_HOST=postgres
OPENAI_API_KEY=your-key-here
```

### `docker-compose.yaml`

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:17
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - postgres

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "8501:8501"
    depends_on:
      - postgres

volumes:
  postgres_data:
  grafana_data:
```

### Running it

```bash
docker-compose up
uv run python db_init.py   # first time / schema changes only
```

- App: `http://localhost:8501`
- Grafana: `http://localhost:3000` (`admin` / `admin`)

```bash
docker-compose down
```

Data in both Postgres and Grafana volumes survives restarts.

> **As shipped, this lesson is a starting point, not a finished stack** —
> the compose file and Dockerfile above exist only in the lesson text, not
> in `code/` (which still runs plain `docker run` for Postgres). `db_init.py`
> still has to be run by hand after `up`. Grafana's data source and
> dashboards are built by hand in the UI (lesson 12), not provisioned from
> files. Turning this into a true one-command stack — provisioned Grafana,
> a Postgres healthcheck, `db_init.py` folded into an init step — is called
> out explicitly as homework in [Lesson 14](#lesson-14--next-steps).

---

## Lesson 14 — Next Steps

Video: [Watch](https://www.youtube.com/watch?v=GpQeAniVGfk&list=PL3MmuxUbc_hLZFNgSad56pDBKK8KO0XIv)

### Recap

RAG pipeline → Streamlit chat UI → every call recorded to PostgreSQL →
dashboards (Streamlit, then Grafana) covering latency/cost/tokens/models →
two quality signals (LLM judge + user thumbs up/down). Net result:
visibility into live behavior, and logs to dig into when something's wrong.

### Build it yourself vs. use a framework

Hand-rolling gives full control over what's captured and where it's stored.
Frameworks trade some of that control for speed:

- [Langfuse](https://langfuse.com/) / [Arize Phoenix](https://phoenix.arize.com/) — LLM app tracing
- [Pydantic Logfire](https://pydantic.dev/logfire) — instruments your code, dashboard comes wired up
- [Evidently](https://www.evidentlyai.com/) — monitoring and evaluation (mostly used for eval)

### Going to production

Two things change at scale:

1. **Overhead** — each call now writes to the DB, plus the judge is another
   LLM call. In production, do this asynchronously (queue + separate
   processing) so it doesn't add latency to the user's request.
2. **Storage** — Postgres is fine for a demo but not ideal for high-volume
   logs. A common pattern: push events to something like Kafka and let
   downstream systems store them.

[OpenTelemetry](https://opentelemetry.io/) is the standard instrumentation
layer underneath tools like Logfire and Langfuse — worth learning even if
you build your own pipeline conceptually similar to this module's.

### Homework — pick one to go further

1. **Monitor an agent** — apply the same instrumentation to an agent from
   the [agents module](https://github.com/DataTalksClub/llm-zoomcamp/tree/main/01-agentic-rag),
   capturing each tool call the way LLM calls were captured here.
2. **Synthetic data + Grafana** — generate data (lesson 11) and watch the
   Grafana dashboard fill out live.
3. **Full Docker Compose** — go past what lesson 13 shows: write out the
   compose file and Dockerfile yourself (they aren't in `code/`), fold
   `db_init.py` into an init step, provision Grafana's data source and
   dashboards from files instead of clicking them together, and add a
   Postgres healthcheck so the app doesn't race the database on cold start.
   Goal: `docker-compose up` is the only command needed.

### Older content

The 2024 cohort used Elasticsearch + Ollama instead of minsearch + OpenAI.
See the [2024 monitoring module](https://github.com/DataTalksClub/llm-zoomcamp/tree/main/cohorts/2024/04-monitoring)
if that setup is more relevant to you.

---

## Full project reference

### File tree (after all lessons)

```text
code/
├── docker-compose.yaml   # lesson 13
├── Dockerfile            # lesson 13
├── .env                  # lesson 13
├── Makefile              # lessons 2, 3, 5
├── pyproject.toml
├── uv.lock
├── .python-version
├── ingest.py              # downloaded, lesson 2
├── rag_helper.py          # downloaded, lesson 2 (RAGBase)
├── evaluation_utils.py    # downloaded, lesson 9
├── assistant.py           # lesson 2, updated in 4/5
├── app.py                 # lesson 3, updated in 4/5/8/9
├── metrics.py             # lesson 4
├── db_init.py             # lesson 5, updated in 8
├── db_save.py             # lesson 5
├── db_query.py            # lesson 6, updated in 7/10
├── dashboard.py           # lesson 7, updated in 10
├── db_feedback.py         # lesson 8
├── judge.py                # lesson 9
└── generate_data.py       # lesson 11
```

### Consolidated Makefile

```makefile
run:
	uv run python assistant.py

chat:
	uv run streamlit run app.py

network:
	docker network create monitoring

postgres: network
	docker run -it \
		--name course-assistant-pg \
		--network monitoring \
		-e POSTGRES_USER=user \
		-e POSTGRES_PASSWORD=password \
		-e POSTGRES_DB=course_assistant \
		-p 5432:5432 \
		-v pgdata:/var/lib/postgresql/data \
		postgres:17
```

### Schema reference

```sql
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    course TEXT NOT NULL,
    model TEXT NOT NULL,
    instructions TEXT NOT NULL,
    prompt TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    response_time FLOAT NOT NULL,
    cost FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id),
    source TEXT NOT NULL,        -- 'user' | 'judge'
    relevance TEXT,               -- judge only
    explanation TEXT,             -- judge only
    score INTEGER,                -- user only, +1 / -1
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
);
```

### Command cheat-sheet

```bash
# Assistant (CLI)
make run
uv run python assistant.py "How do I join the course?"

# Chat app
make chat                                                # :8501

# Dashboard
uv run streamlit run dashboard.py --server.port 8502     # :8502

# Postgres
make postgres
uv run python db_init.py

# Inspect data
docker exec -it course-assistant-pg psql -U user -d course_assistant \
    -c "SELECT id, question, response_time, cost FROM conversations;"

# Judge (standalone test)
uv run python judge.py

# Synthetic traffic
uv run python generate_data.py

# Grafana
docker run -d --name grafana --network monitoring -p 3000:3000 \
    -v grafana_data:/var/lib/grafana grafana/grafana         # :3000

# Full stack
docker-compose up
docker-compose down
```

### Ports

| Port | Service |
|---|---|
| 8501 | Streamlit chat app |
| 8502 | Streamlit dashboard |
| 5432 | PostgreSQL |
| 3000 | Grafana |

---

*Reference README covering all 14 lessons of DataTalksClub LLM Zoomcamp
Module 5 — Monitoring. Source: [05-monitoring/lessons](https://github.com/DataTalksClub/llm-zoomcamp/tree/main/05-monitoring/lessons).*
