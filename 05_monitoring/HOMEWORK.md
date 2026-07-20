# Homework: Monitoring

In module 5 we learned how to monitor our RAG system: capture metrics from each LLM call, store them in a database, and visualize them on a dashboard.

In the module we built all of this by hand - a custom dataclass for the metrics, PostgreSQL for storage, Streamlit and Grafana for dashboards.

In this homework, we will explore an alternative: OpenTelemetry (OTel). This is the industry standard for code instrumentation. Every monitoring framework we mentioned is built on top of it - like Logfire, Langfuse, Arize Phoenix and others.

In this homework we will use OTel directly. We will instrument our RAG with traces, capture metrics as span attributes, persist the spans to SQLite, and build a dashboard from the trace data.

We keep using the same course-lessons RAG from homework 1. The knowledge base is the 72 lesson pages pulled from GitHub, indexed with minsearch.

It's possible your answers won't match exactly. If so, select the closest one.

# Setup
Create a fresh project:

```bash
mkdir llm-zoomcamp-hw5 && cd llm-zoomcamp-hw5
uv init
uv add gitsource minsearch openai python-dotenv
```

We want everyone to start with the same code, so we prepared a starter package.

Download it:

```bash
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/cohorts/2026/05-monitoring
wget $PREFIX/rag_helper.py
wget $PREFIX/starter.py
```

We keep things simpler and focus only on RAG. However, all the concepts could be directly translated to agents.

Next, you need to put your OpenAI key in a .env file:

```bash
OPENAI_API_KEY=sk-...
```

Like previously, you can use any alternative you want.

The starter loads the 72 course lessons, builds a text-search index, and wraps it in a RAGBase instance you can call right away:

```python
from starter import rag

query = "How does the agentic loop keep calling the model until it stops?"
answer = rag.rag(query)
print(answer)
```

For the LLM, we recommend OpenAI with gpt-5.4-mini, but you can use any model and provider you want.

# OpenTelemetry setup
First, install the OpenTelemetry libraries:

```bash
uv add opentelemetry-api opentelemetry-sdk
```

- opentelemetry-api is the interface - the classes and functions you import in your code (trace, Tracer, Span)
- opentelemetry-sdk is the implementation that actually creates and processes spans.

# OpenTelemetry
Before we start, we need to learn a few concepts from OTel - we will use them in this homework.

- A trace is the end-to-end story of a single request as it moves through your system. For us, it's one RAG call.
- A span is one operation within a trace. A trace is made of one or more spans, organized as a tree. Each span has a name, a start and end time, and a set of attributes. For us we will have one span inside the trace, but for agents one trace will have multiple spans.
- Attributes are key-value pairs attached to a span - anything you want to record, like the number of tokens used or the cost of a call.

When a span finishes - meaning the code block it wraps completes - the SDK hands it to a span processor, which forwards it to an exporter. The exporter decides where the span goes: to the console, to a file, to a database, or to a remote collector. We will see all of this in practice in the questions below.

We start with the ConsoleSpanExporter, which prints each finished span to the terminal so we can see what OTel captures:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")
```

Here is what each line does:

- TracerProvider() creates the SDK's central configuration object. It owns the span processors and decides how spans are built.
- SimpleSpanProcessor(ConsoleSpanExporter()) wires a processor that forwards every finished span to the console exporter, one at a time. "Simple" means synchronous and immediate - good for development.
- trace.set_tracer_provider(provider) registers the provider globally, so every call to trace.get_tracer(...) returns a tracer backed by it.
- trace.get_tracer("llm-zoomcamp") returns a Tracer we use to create spans. The string is just a label for the instrumentation scope - it identifies which part of the code produced the spans.

Put this block at the top of your script, before you import or use starter - so the tracer provider is ready before any code that might create spans.

With the tracer in hand, you can wrap any block of code in a span:

```python
with tracer.start_as_current_span("my_operation") as span:
    # your code here
    span.set_attribute("my_key", "my_value")
```

start_as_current_span creates a new span and makes it the "current" span for the duration of the with block. Any code inside the block - including other calls to start_as_current_span - becomes a child of this span. When the block exits, the span ends automatically.

You will use this pattern to instrument the RAG methods in the questions below.

# Q1. First trace
Wrap the rag() method so each call produces a span. The simplest way is to create a RAGTraced subclass of RAGBase that wraps rag(), search(), and llm() each in their own span.

Run this query:

`How does the agentic loop keep calling the model until it stops?`

The console exporter prints every finished span as a dictionary. Count the spans in the console output - each one is a separate ReadableSpan entry. How many spans does the trace produce?

- 1
- 3 <-- answer
- 5
- 7

## The concept

Looking at rag_helper.py's RAGBase.rag():

```python
def rag(self, query):
    search_results = self.search(query)      # 1 call
    prompt = self.build_prompt(query, search_results)  # not wrapped
    response = self.llm(prompt)               # 1 call
    return response.output_text
```

`rag()` calls `search()` once and `llm()` once — `build_prompt()` isn't in the list of methods you're told to wrap. So RAGTraced needs to wrap exactly `rag`, `search`, and `llm`, each in its own span, with `rag`'s span as the
parent (since it's the outer with block) and search/llm's spans as
children (created while rag's span is still current).

Here's the full script — save it as e.g. `q1_trace.py` inside your
`llm-zoomcamp-hw5` project, alongside `starter.py` and `rag_helper.py`:

Run `uv run python q1_trace.py`

# Output

```bash
05_monitoring % uv run python q1_trace.py
{
    "name": "search",
    "context": {
        "trace_id": "0x517407040e891933845bd3ff86668979",
        "span_id": "0xddaab819e08fdc4c",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": "0xad6c73772808c71a",
    "start_time": "2026-07-18T22:16:09.930660Z",
    "end_time": "2026-07-18T22:16:09.931484Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {},
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.44.0",
            "service.instance.id": "12594832-6ee7-4971-9599-ea5812aa288c",
            "service.name": "unknown_service"
        },
        "schema_url": ""
    }
}
{
    "name": "llm",
    "context": {
        "trace_id": "0x517407040e891933845bd3ff86668979",
        "span_id": "0x4de78ace9fafa240",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": "0xad6c73772808c71a",
    "start_time": "2026-07-18T22:16:09.931658Z",
    "end_time": "2026-07-18T22:16:13.865794Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {},
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.44.0",
            "service.instance.id": "12594832-6ee7-4971-9599-ea5812aa288c",
            "service.name": "unknown_service"
        },
        "schema_url": ""
    }
}
{
    "name": "rag",
    "context": {
        "trace_id": "0x517407040e891933845bd3ff86668979",
        "span_id": "0xad6c73772808c71a",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": null,
    "start_time": "2026-07-18T22:16:09.930622Z",
    "end_time": "2026-07-18T22:16:13.866060Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {},
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.44.0",
            "service.instance.id": "12594832-6ee7-4971-9599-ea5812aa288c",
            "service.name": "unknown_service"
        },
        "schema_url": ""
    }
}
It keeps calling the model inside a `while True` loop.

Each iteration:
1. Sends the full `messages` history to the model.
2. Checks the response for any `function_call` items.
3. Runs those tools and appends the outputs to `messages`.
4. If there were no function calls, it `break`s out of the loop.

So the stop condition is:

- **no function calls in the model’s response** → done

The key flag is `has_function_calls`. If it stays `False` for an iteration, the loop ends.
```

## Trace details 

All three spans share the same trace_id
(0x517407...8979) — that's what makes them one trace, not three unrelated
events. And look at parent_id:

| **Span** | **Span ID** | **Parent ID** | **Description** |
|----------|-------------|---------------|-----------------|
| `rag` | `0xad6c7377...c71a` | `null` | Root span representing the entire RAG workflow. Since it has no parent, it is the top-level operation. |
| `search` | `0xddaab819...dc4c` | `0xad6c7377...c71a` | Child span of `rag` that captures the document retrieval/search phase. |
| `llm` | `0x4de78ace...a240` | `0xad6c7377...c71a` | Child span of `rag` that captures the LLM inference/generation phase. |

search and llm both point their parent_id straight at rag's
span_id — that's the parent/child link made concrete,

## What the timestamps tell you

search:  09.930660 → 09.931484   (~0.8 ms)
llm:     09.931658 → 13.865794   (~3.93 s)
rag:     09.930622 → 13.866060   (~3.94 s, ≈ search + llm)

This is the part that makes tracing actually useful, beyond just answering
Q1: search (hitting the in-memory minsearch index) is essentially free — under a millisecond. Practically all of your ~3.94s response time is the llm span, i.e. the OpenAI network call. rag's own duration is just the sum of its children plus negligible overhead from build_prompt/build_context in between. If you were debugging "why is this endpoint slow," this breakdown immediately tells you where to look — and it's why real observability tools build dashboards on exactly this kind of span duration data.

One more thing worth noticing for later: "service.name": "unknown_service" under resource.attributes — that's OTel's default when you don't configure a service name. 

# Q2. Capturing metrics as span attributes
Spans are not just timing markers - you can attach any information you want to them with set_attribute. We already use spans to record how long each step takes. Now we'll add the metrics we care about: tokens and cost.

Read the token usage from the LLM response (the llm() method in the starter already returns the raw response object) and set them as attributes on the llm span:

```python
span.set_attribute("input_tokens", usage.input_tokens)
span.set_attribute("output_tokens", usage.output_tokens)
```

And since we know both input and output tokens, we can also compute the cost using the code from the previous modules.

Now re-run the query. How many input tokens do we see?

- 700
- 7000 <-- answer
- 70000
- 700000

These numbers vary between runs. Pick the closest option.

## The concept

Q1 gave you timing for free — a span's start_time/end_time are
automatic. Q2 adds your own data to a span via span.set_attribute(key, value). Attributes are just a dict that rides along with the span through
the exporter — in the console output you saw in Q1, that's the (currently
empty) "attributes": {} field.

Two things matter about set_attribute:

- It only works while the span is still open — you need a reference to the span object, which means changing with tracer.start_as_current_span("llm"): to with tracer.start_as_current_span("llm") as span:.
- The values you attach have to come from inside that same block, because that's the only place you have access to the raw LLM response (and therefore response.usage) before RAGBase.llm() hands back just the response object.

## Solution:
Only the llm method of RAGTraced changes from Q1 — rag and search
stay as they were:

```python
def calculate_cost(model, usage):
    # same per-million-token pricing pattern used in lesson 4 (metrics.py)
    cost = 0
    if "gpt-5.4-mini" in model:
        cost = (usage.input_tokens * 0.15 + usage.output_tokens * 0.60) / 1_000_000
    return cost


class RAGTraced(RAGBase):

    def rag(self, query):
        with tracer.start_as_current_span("rag"):
            return super().rag(query)

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search"):
            return super().search(query, num_results=num_results)

    def llm(self, prompt):
        # capture the span reference with `as span` so we can call
        # set_attribute on it before the `with` block closes
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)   # raw OpenAI response object
            usage = response.usage

            # attach the numbers we care about to THIS span
            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)

            cost = calculate_cost(self.model, usage)
            span.set_attribute("cost", cost)

            return response   # still returns the response, unchanged for rag()

```

Run `uv run python q2_capture_metric_as_span_attributes.py`

## Output
```bash
05_monitoring % uv run python q2_capture_metric_as_span_attributes.py

{
    "name": "search",
    "context": {
        "trace_id": "0xbb092edb922eccf8edc557a85f3b2542",
        "span_id": "0x91666136d5695089",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": "0x00736a2fdbee3c62",
    "start_time": "2026-07-18T23:08:02.510628Z",
    "end_time": "2026-07-18T23:08:02.512100Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {},
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.44.0",
            "service.instance.id": "039f2abb-f6a8-4539-b88a-9ea2f10e6d87",
            "service.name": "unknown_service"
        },
        "schema_url": ""
    }
}
{
    "name": "llm",
    "context": {
        "trace_id": "0xbb092edb922eccf8edc557a85f3b2542",
        "span_id": "0xbe05615a38dd3bee",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": "0x00736a2fdbee3c62",
    "start_time": "2026-07-18T23:08:02.512261Z",
    "end_time": "2026-07-18T23:08:04.947986Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {
        "input_tokens": 7111,
        "output_tokens": 117,
        "cost": 0.00113685
    },
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.44.0",
            "service.instance.id": "039f2abb-f6a8-4539-b88a-9ea2f10e6d87",
            "service.name": "unknown_service"
        },
        "schema_url": ""
    }
}
{
    "name": "rag",
    "context": {
        "trace_id": "0xbb092edb922eccf8edc557a85f3b2542",
        "span_id": "0x00736a2fdbee3c62",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": null,
    "start_time": "2026-07-18T23:08:02.510595Z",
    "end_time": "2026-07-18T23:08:04.951915Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {},
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.44.0",
            "service.instance.id": "039f2abb-f6a8-4539-b88a-9ea2f10e6d87",
            "service.name": "unknown_service"
        },
        "schema_url": ""
    }
}
The loop keeps calling the model with a `while True` loop.

Each turn it:
1. sends the full `messages` history to the model,
2. checks the response for any `function_call` items,
3. runs those tools and appends the results,
4. and then calls the model again.

It stops when a response comes back with no function calls. That sets `has_function_calls` to `False`, and the code does `break`.

So the exit condition is: **no tool calls this turn means the agent is done**.
```

## Reasoning about the answer
input_tokens is the size of the full prompt sent to the model —
instructions + your question + the retrieved context, which for this RAG is 5 full lesson pages (num_results=5 in RAGBase.search) concatenated verbatim by build_context(). Looking at the actual lesson file sizes on disk (we downloaded all of them earlier — they range roughly 1.7KB to 9.5KB, averaging somewhere around 3–4KB each), 5 of them stacked together is roughly 15–20KB of raw text. At the rough rule-of-thumb of ~4 characters per token, that lands around 4,000–5,000 tokens just for context, plus the
instructions and question on top.

That's closest to the 7000 option — well above 700 (way too small for 5 full lesson pages) and nowhere near 70000 or 700000 (that would be more like 50+ lesson pages, not 5). But token counts depend on exactly which 5 lessons minsearch retrieves for this specific query, so treat this as an estimate — run it yourself and read the actual input_tokens value out of the printed span to confirm.

Worth noting from the same span while we're looking at it: output_tokens: 117 and cost: 0.00113685 — a single call to gpt-5.4-mini on this RAG costs about a tenth of a cent, almost entirely driven by the 7111 input tokens (at $0.15/M) rather than the 117 output tokens (at $0.60/M). That `asymmetry — RAG systems are usually input-token-dominated because of the retrieved context — is exactly the kind of thing this attribute capture makes visible that raw latency numbers wouldn't show you`.

# Q3. Span timing
Each span automatically records its duration. Look at the console output from Q1 and find the durations for the search span and the llm span.

For a typical query, roughly how long does the LLM call take?

- Under 100ms
- 100-500ms
- 500-2000ms
- Over 2000ms <-- answer

The first call can be slower (cold start). Pick the range you see most often.

## The concept

From Q1's output alone:


search:  09.930660Z → 09.931484Z   →  0.824 ms
llm:     09.931658Z → 13.865794Z   →  3934.1 ms
llm took ~3934ms, well past the 2000ms line, search was under 1ms.

Answer: Over 2000ms

# Q4. Saving traces to SQLite
Right now the spans are printed to the terminal and then gone. We don't save them.

We want to persist them so we can query them later.

In this homework, we'll use SQLite - it's a more lightweight option than Postgres, so we don't need to set up any docker containers in this homework.

Our instrumentation is already done, we don't need to change anything there. But we need to create a custom exporter. Instead of printing the spans, it will save them to the database.

OTel calls the exporter through the same span processor we already use, we just swap the destination.

Now we will create a custom exporter that saves each finished span to a SQLite database. The exporter extends SpanExporter. It has the following methods:

- export method that receives a list of ReadableSpan objects
- shutdown and force_flush methods

Let's implement it:

```python
import sqlite3
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True
```

Replace the console exporter with this new exporter:

```python
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)
```

Re-run the query from Q1. Which span names appear in the spans table?

- Only rag
- rag and llm
- rag, search, and llm <-- answer
- search, llm, and judge

## The concept
Nothing about what spans get created changes here — the instrumentation (RAGTraced.rag/search/llm, each wrapped in start_as_current_span) from Q1–Q2 is untouched. Only the destination changes: you're swapping which SpanExporter implementation the SimpleSpanProcessor hands finished spans to.ConsoleSpanExporter.export() calls print(); SQLiteSpanExporter.export() runs an INSERT. Same pipeline, same three spans, different sink.

This is the actual point of the exporter abstraction in OTel: your
instrumentation code (the with tracer.start_as_current_span(...) blocks inside RAGTraced) never has to know or care where spans end up. You could point it at Postgres, a file, or a remote collector without touching RAGTraced at all.

## What the exporter does, piece by piece

```python
class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path="traces.db"):
        # opens/creates the SQLite file, and creates the table
        # if this is the first run (IF NOT EXISTS)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,   -- OTel timestamps are nanoseconds since epoch, stored raw
                end_time INTEGER,
                input_tokens INTEGER, -- NULL for spans that never call set_attribute on these
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        # SimpleSpanProcessor calls this once per finished span (synchronously,
        # right when the `with` block closes) — `spans` is a short list,
        # usually just the one span that just ended
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    # .get() returns None for spans that never set these —
                    # i.e. "rag" and "search", which only "llm" populates
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS   # tells OTel the export didn't fail

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True   # no buffering happening here, so nothing to flush
```

## Run
```bash
05_monitoring % uv run python q4_saving_trace_to_sqlite.py
# Output
It keeps calling the model in a `while True` loop.

Each iteration:
1. Send the full `messages` history to the model.
2. Check the response for any `function_call` items.
3. If there are tool calls, run them and append the results to `messages`.
4. If there are no function calls, `break` out of the loop.

So the stop condition is:

- **no function calls in the latest response**

That means the model has given its final answer.
```

```bash
05_monitoring % sqlite3 traces.db "SELECT name, start_time, end_time, input_tokens, output_tokens, cost FROM spans;"
search|1784507616657488000|1784507616659024000|||
llm|1784507616660166000|1784507619511286000|7111|107|0.00113085
rag|1784507616657442000|1784507619512234000|||
```

## Why the answer is "rag, search, and llm"
export() is called once per span that finishes, and exactly the same three spans finish as in Q1 — search (child), llm (child), rag
(parent). 

# Q5. Querying trace data
The traces are now in SQLite. Run one more query through the traced RAG, then query the database.

The rag span wraps everything, so its duration includes both search and llm. To see where time actually goes, exclude the rag span and compare the children.

Using SQL (or pandas), compute the total duration for each span name excluding rag. Which span type takes the most total time?

- search
- llm <-- answer
- They're all about the same

## The concept
The question wants you to run the traced RAG once more (so the table has at least two full traces — 6 rows: two search, two llm, two rag), then aggregate durations grouped by span name, excluding rag. The "exclude rag" instruction matters: rag's duration is the sum of search + llm (plus negligible glue code) since it wraps both — including it would just double-count time already captured by its children, and it'd trivially "win" any total-duration comparison for no interesting reason.

## Running it again + querying
Re-run the same script from Q4 (with the SQLite exporter) a second time so you accumulate more rows in traces.db:

```bash
05_monitoring % uv run python q4_saving_trace_to_sqlite.py

# Output
It keeps calling the model inside a `while True` loop.

Each iteration:
- sends the full `messages` history to the model,
- checks the response for any `function_call` items,
- runs those tools and appends the results,
- sets `has_function_calls = True` if any tool was called.

At the end of the turn, if `has_function_calls == False`, it breaks out of the loop. So the loop stops when the model returns a response with no function calls, meaning it has a final answer.
```

## Then aggregate with SQL:
```sql
SELECT
    name,
    SUM(end_time - start_time) AS total_duration_ns
FROM spans
WHERE name != 'rag'
GROUP BY name
ORDER BY total_duration_ns DESC;
```

```bash
05_monitoring % sqlite3 traces.db "
SELECT name, SUM(end_time - start_time) AS total_duration_ns
FROM spans
WHERE name != 'rag'
GROUP BY name
ORDER BY total_duration_ns DESC;
"
llm|5660352000
search|2839000
```

llm total: 5,660,352,000 ns (~5.66s) vs search total: 2,839,000 ns (~2.8ms), a gap of roughly 1,994x. 

Across your accumulated runs, llm completely dominates total span time, exactly as the single-trace numbers predicted.

Answer: llm

# Q6. Token stability across runs
Load the SQLite data with pandas. One thing a dashboard can tell you is how stable your system is. If the same query always produces the same number of input tokens, the context your RAG retrieves is consistent. If it varies a lot, something in the search may be unstable.

Run the same query from Q1 three more times (so you have 4 RAG calls total in the database). Then compute the input tokens for each llm span.

How much do the input tokens vary across these 4 runs?

- They're identical <-- answer
- Within 10% of each other
- Within 50% of each other
- They vary more than 50%

## The concept

This question is checking something subtle: `input_tokens` depends
entirely on the text you send to the model — instructions + question + retrieved context. Nothing about that text is randomized. Unlike the LLM's output (which samples from a distribution, so `output_tokens` naturally varies run to run — you already saw this: 117 in Q2, 107 in Q4, same query both times), the input side has no randomness anywhere in the pipeline:

- `search()` uses minsearch's keyword/TF-IDF-style scoring against a fixed, already-built index — same query text always ranks the same documents in the same order, deterministically.
- `build_context() / build_prompt()` are pure string formatting — same inputs, same output string, every time.
- OpenAI's tokenizer is deterministic — the same text always encodes to the same token count.

So unless minsearch has ties that break inconsistently, or you accidentally change num_results/the query text between runs, input_tokens should be essentially frozen across repeated calls with the same query.

## Running it 2 more times + loading with pandas

```bash
uv run python q4_saving_trace_to_sqlite.py
uv run python q4_saving_trace_to_sqlite.py
```

That gives you 4 total RAG calls in traces.db (2 from earlier + 2 more).
Load and inspect with pandas:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("traces.db")
df = pd.read_sql("SELECT * FROM spans", conn)

llm_spans = df[df.name == "llm"].reset_index(drop=True)
print(llm_spans[["input_tokens", "output_tokens", "cost"]])

variation = (
    (llm_spans.input_tokens.max() - llm_spans.input_tokens.min())
    / llm_spans.input_tokens.mean()
)
print(f"input_tokens range as % of mean: {variation:.2%}")
```

## Run `uv run python q6_token_stability_across_runs.py`

## Output
```bash
05_monitoring % uv run python q6_token_stability_across_runs.py
   input_tokens  output_tokens      cost
0        7111.0          100.0  0.001127
1        7111.0          105.0  0.001130
2        7111.0          110.0  0.001133
3        7111.0          121.0  0.001139
```

### All four input_tokens values are identical — 7111 across every run — while output_tokens drifts (100 → 105 → 110 → 121), exactly the pattern predicted: deterministic retrieval/prompt construction on the input side, stochastic generation on the output side.

## Answer: They're identical

## What you should already expect from your own data
You don't have to guess blind here — you've already run this exact query twice, in Q2 and in Q4, and both times input_tokens came back as 7111, identical to the token, even though output_tokens drifted (117 → 107) between those same two runs. That's the pattern predicted above playing out already: input side frozen, output side wobbling. Two more runs should just reproduce that same 7111 a couple more times (or, in the rare case minsearch has a scoring tie among your 72 lesson docs that resolves differently, it might shift by one retrieved document — but that would be an all-or-nothing jump to a very different token count, not a
gentle drift, and would show up as an obvious outlier in your 4 numbers, not a general trend).

Answer: They're identical