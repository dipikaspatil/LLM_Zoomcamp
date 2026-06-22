# Module 01 — Agentic RAG: Concepts & Learnings

## What is RAG?

**Retrieval-Augmented Generation (RAG)** combines search with LLM text generation to ground answers in real documents.

**Why LLMs alone aren't enough:**
- Knowledge cutoff — no awareness of recent or private data
- No access to internal/proprietary documents
- Hallucinations — confident but incorrect answers

**RAG solves this** by retrieving relevant documents at query time and including them in the prompt.

```
User Query → Search → Retrieve Top-K Docs → Build Prompt → LLM → Answer
```

> "Your model is only as good as your retrieval, so search quality matters a lot for RAG."

The three components (search, prompt building, LLM) are **independent and swappable** — you can change the search backend or LLM without touching the rest.

---

## Search with minsearch

Lightweight in-memory search engine — same concepts as Elasticsearch, runs anywhere Python runs.

```python
import minsearch

index = minsearch.Index(
    text_fields=["content"],    # tokenized, ranked by TF-IDF relevance
    keyword_fields=["filename"] # exact-match filtering only
)
index.fit(documents)

# Full-text search across all documents
results = index.search("agentic loop")

# Filter to a specific file
results = index.search("agentic loop", filter_dict={"filename": "14-agentic-loop.md"})
```

| Field Type | How It Works | Use For |
|---|---|---|
| `text_fields` | Tokenized, scored by relevance | Free-form content, prose |
| `keyword_fields` | Exact match only | IDs, filenames, categories |

All search engines share the same principle: score every document for similarity to the query and return the top results. The difference is how similarity is defined (keyword vs. semantic).

---

## Prompt Engineering

A well-structured prompt has two distinct parts:

- **Instructions (fixed)** — tell the LLM how to behave; defined once, reused across requests
- **User prompt (dynamic)** — the question + retrieved context; changes every request

```python
INSTRUCTIONS = """
Answer questions based on the provided content.
If the answer is not in the context, respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTENT:
{content}
""".strip()
```

**Key principles:**
- Instructions reduce hallucinations by constraining the LLM to the retrieved context
- Prompt engineering is part art, part science — experiment and measure, don't guess
- The prompt is the bridge between search and the LLM; quality here directly affects output quality

---

## LLM Integration — OpenAI Responses API

The course uses the **Responses API** (newer than Chat Completions). Many providers offer an OpenAI-compatible client.

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.4-mini",
    input=[
        {"role": "developer", "content": INSTRUCTIONS},
        {"role": "user", "content": prompt}
    ]
)

answer = response.output_text  # shortcut for response.output[0].content[0].text
tokens_used = response.usage.input_tokens
```

**Message roles:**
- `developer` — system-level instructions, consistent across all requests
- `user` — the changing user question + context

**Cost awareness:** Track `response.usage.input_tokens` — the prompt size is the same regardless of provider, making token count a provider-neutral metric for measuring efficiency.

---

## Data Ingestion — Separation of Concerns

Avoid loading and indexing data at every startup. Separate the two processes:

```
Ingestion process  →  index documents once  →  persistent storage
Query process      →  open existing index   →  search as often as needed
```

- **In-memory (minsearch):** simple, but re-indexes on every restart — fine for development
- **Persistent (sqlitesearch, Elasticsearch, Qdrant):** survives restarts — required for production

SQLite-backed search (`sqlitesearch`) keeps the same API as minsearch but persists to disk — no extra dependencies since SQLite ships with Python.

---

## Chunking

Long documents hurt retrieval precision — a match deep inside a page pulls the entire page into the prompt. Chunking splits documents into smaller overlapping pieces.

```python
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
```

**How the sliding window works:**
- Each chunk is `size` characters of the original page
- Window moves forward by `step` characters
- Consecutive chunks **overlap** by `size - step` characters — passages near boundaries appear whole in at least one chunk
- Each chunk inherits `filename` and gains a `start` offset

**Benefits:**
- Smaller, more focused context sent to the LLM → fewer input tokens
- Better retrieval — the top-K chunks are more likely to be entirely relevant, vs. a full page where the answer appears in just one section

---

## From RAG to Agents

Plain RAG is a **fixed pipeline** — the developer decides the flow up front: search runs once, with the exact user query, no matter what.

**What agents add:** The LLM decides when to search, what to search for, and how many times.

```
RAG:   User Query → [search once] → LLM → Answer
Agent: User Query → LLM → [search?] → LLM → [search again?] → LLM → Answer
```

**Why agents are more powerful:**
- Can retry with reformulated queries when initial results are poor
- Can correct user typos (e.g., "Olama" → "Ollama") without being explicitly programmed to
- Can draw from multiple searches before answering
- Trade-off: multiple LLM calls = higher latency and cost

---

## Function Calling

Mechanism that lets the LLM request tool execution rather than generating text directly.

**How it works:**
1. Define tools as JSON schemas describing name, purpose, and parameters
2. Include tools in every API request
3. When the model wants to search, it returns a `function_call` item instead of text
4. You execute the function and send results back as a message
5. The model then decides its next step

```python
tools = [{
    "type": "function",
    "name": "search",
    "description": "Search the course content for relevant information.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}]
```

The JSON schema is what the model reads to understand what the tool does and what arguments it expects. A good description and parameter names directly influence how well the model uses the tool.

---

## The Agentic Loop

The core pattern behind every agent:

```python
while True:
    response = llm(messages, tools)

    function_calls = [item for item in response.output if item.type == "function_call"]

    if not function_calls:
        break  # model is done — final answer

    messages += response.output  # append assistant output to history

    for fc in function_calls:
        result = execute_tool(fc)
        messages.append({"type": "function_call_output", "call_id": fc.call_id, "output": result})
```

**Three essential components:**

| Component | Role |
|---|---|
| **Instructions** | Define agent behavior — the primary lever for steering the agent |
| **Tools** | Functions the agent can invoke |
| **Memory** | Full message history — lets the agent track what it has tried |

**Instructions are powerful:** You can instruct the agent to make multiple searches, restrict its domain, or recover from poor results — without changing any code.

---

## Agent Frameworks

Frameworks eliminate boilerplate by wrapping the agentic loop pattern.

**What frameworks do for you:**
- Auto-generate tool schemas from Python type hints and docstrings
- Manage message history across turns
- Track token usage and cost
- Handle the while-loop and tool dispatch

```python
# Without framework — write schema manually + manage the loop
tools = [{"type": "function", "name": "search", "description": "...", "parameters": {...}}]
while True:
    ...

# With toyaikit — schema auto-generated, loop handled internally
from toyaikit.tools import Tools
from toyaikit.llm import OpenAIClient
from toyaikit.chat.runners import OpenAIResponsesRunner

tools = Tools()
tools.add_tool(search_fn)  # reads type hints + docstring automatically

runner = OpenAIResponsesRunner(tools=tools, developer_prompt=INSTRUCTIONS, llm_client=...)
result = runner.loop(question)
```

**Available frameworks** (all share the same core patterns):
- **toyaikit** — minimal, educational, transparent
- **OpenAI Agents SDK** — official, production-grade
- **PydanticAI** — type-safe, Pydantic-native
- **LangChain** — ecosystem with many integrations

> "Every modern agent framework does the same trick — reads a typed Python function with a docstring and builds the schema from it."

The pattern you learn in one framework transfers directly to the others.
