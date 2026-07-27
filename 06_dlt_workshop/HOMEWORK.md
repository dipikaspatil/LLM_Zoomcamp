# Homework: dlt
In this homework we will take the FAQ agent from Module 1, instrument it with Pydantic Logfire for observability, then pull the trace data back out with dlt and analyze it.

In Module 1 we wrote the agent loop by hand and then we saw toyaikit - an agentic framework.

For this homework we rewrote into Pydantic AI, so it's easier to integrate it with Logfire. Pydantic AI [https://pydantic.dev/docs/ai/overview/] and Logfire work really well together, that's why we use them here.

In Module 5 we learn about monitoring and observability, and implement our own monitoring solution. Logfire is an alternative for that.

# Getting the code
The rewritten agent is in the homework/ directory. Download it with wget:

```bash
python3 -m venv .venv &&  source .venv/bin/activate

PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/cohorts/2026/workshops/dlt/homework

wget $PREFIX/agent.py
wget $PREFIX/ingest.py
wget $PREFIX/main.py
wget $PREFIX/.env.example -O .env
```

The agent code is in homework/agent.py [https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/workshops/dlt/homework/agent.py]. Here we use Pydantic AI which we didn't cover previously. Conceptually there's nothing new: we covered everything already in module 1. The comments in the file will explain what's happening and how things map to what we learned previously.

Make sure to read through it before proceeding.

# Setup
We start by configuring our project:

```python
uv init
uv add openai minsearch requests python-dotenv pydantic-ai logfire
uv add "dlt[duckdb]"
```

Open .env and add your OPENAI_API_KEY:

```python
OPENAI_API_KEY=sk-YOUR_KEY_HERE
```

Make sure it's in .gitignore:
```python
.env
```

You can use any other provider instead of OpenAI. Check Pydantic AI documentation to see how you can use your provider.

Verify that the agent runs:
```bash
uv run python main.py
```

# Question 1. Instrument the agent with Logfire
Sign up for a free Logfire account, create a project, and generate a write token. Put it in .env as LOGFIRE_TOKEN.

Instrument the agent:
```python
logfire.configure()
logfire.instrument_pydantic_ai()
```

Run the agent a few times with different questions and open your project on Logfire to see the traces.

For the following query

`How do I run Ollama locally?`

how many spans does a single agent run produce?

Each span is either the agent run itself, an LLM call, or a tool call. The number can vary between runs because the model decides how many times to search.

- 1
- 5 <-- answer
- 15
- 30

## Output
```bash
06_dlt_workshop % uv run python main.py
Logfire project URL: https://logfire-us.pydantic.dev/dipika-s-patil/starter-project
15:52:51.811 faq_agent run
15:52:51.827   chat gpt-5.4-mini
15:52:54.901   running tool: search
15:52:54.913   chat gpt-5.4-mini
To run Ollama locally, the FAQ says:

1. Install Ollama from **https://ollama.com/download** for your OS:
   - **macOS**: download and install the `.pkg`
   - **Windows**: download and install the `.msi`
   - **Linux**: run:
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ```

2. Start a model locally:
   ```bash
   ollama run llama3
   ```
   This downloads the model and opens a local chat interface.

3. If you want to check the local server:
   ```bash
   curl http://localhost:11434
   ```

4. If you want to use it from Python:
   ```bash
   pip install ollama
   ```

   Example:
   ```python
   import ollama

   response = ollama.chat(
       model='llama3',
       messages=[{"role": "user", "content": your_prompt}]
   )

   print(response['message']['content'])
   ```

If you want, I can also help with the course-specific setup for Ollama or other areas you want to explore.
```

```bash
faq_agent run                 ← span 1: the agent run (root span)
  chat gpt-5.4-mini            ← span 2: LLM call #1 (decides to search)
  running tool: search         ← span 3: tool call (search executes)
  chat gpt-5.4-mini            ← span 4: LLM call #2 (writes the final answer)
```

Answer - 5

# Question 2. Load traces into DuckDB with dlt
Generate a read token for your Logfire project and set it as LOGFIRE_READ_TOKEN in .env.

Initialize a dlt-hub project like in the workshop. Then ask your coding agent to pull the data from Pydantic Logfire and save it into DuckDB.

The dltHub AI workbench has a ready-made context for Logfire. Point your agent to it: https://dlthub.com/context/source/logfire

If you don't currently use a coding agent, you can use something like OpenCode: you should be able to complete one session with the free account.

Alternatively, you can do it in the old way (using ChatGPT or your favorite search engine).

If you don't currently use a coding agent, you can use something like OpenCode: you should be able to complete one session with the free account.

Alternatively, you can do it in the old way (using ChatGPT or your favorite search engine).

The logfire traces contain deeply nested JSON (span attributes with LLM messages, tool calls, token usage, etc.). dlt automatically normalizes this into a set of tables - one for the main records, plus child tables for each nested level.

How many tables did dlt create? Check with:

```sql
SELECT COUNT(*) FROM information_schema.tables 
WHERE table_schema = 'agent_traces';
```

- 1
- 3
- 24 <-- answer
- 100

## Steps followed - 
1. Generate a read token for the same Logfire project, save as LOGFIRE_READ_TOKEN in .env.

2. Scaffold a dltHub workspace exactly like the workshop:
```bash
uvx dlthub-init@latest
```

3. Point your coding agent at the ready-made Logfire context and ask it to build the pipeline:

pull my traces from Pydantic Logfire (see https://dlthub.com/context/source/logfire) using LOGFIRE_READ_TOKEN, and load them into DuckDB

The agent will install a Logfire-aware toolkit/skill the same way it auto-installed filesystem-pipeline and rest-api-pipeline in the workshop — it reads the linked context page to learn Logfire's API shape (it exposes trace/span data, typically via an OTLP-compatible export or query API) and scaffolds a pipeline against it.

Output

```bash
 06_dlt_workshop % uv run python q2_load_traces_to_duckdb.py
2026-07-27 16:17:28,541|[WARNING]|49961|8400706944|06_dlt_workshop|validate.py|verify_normalized_table:113|In schema `logfire_ingest`: The following columns in table 'spans' did not receive any data during this load and therefore could not have their types inferred:
  - attributes__model_request_parameters__output_object
  - attributes__model_request_parameters__prompted_output_template
  - attributes__model_request_parameters__thinking
  - deployment_environment
  - exception_message
  - exception_stacktrace
  - exception_type
  - http_method
  - http_response_status_code
  - http_route
  - log_body
  - otel_status_message
  - url_full
  - url_path
  - url_query

Unless type hints are provided, these columns will not be materialized in the destination.
One way to provide type hints is to use the 'columns' argument in the '@dlt.resource' decorator.  For example:

@dlt.resource(columns={'attributes__model_request_parameters__output_object': {'data_type': 'text'}})

2026-07-27 16:17:28,541|[WARNING]|49961|8400706944|06_dlt_workshop|validate.py|verify_normalized_table:113|In schema `logfire_ingest`: The following columns in table 'spans__attributes__model_request_parameters__function_tools' did not receive any data during this load and therefore could not have their types inferred:
  - capability_id
  - include_return_schema
  - metadata
  - outer_typed_dict_key
  - return_schema
  - timeout
  - tool_kind
  - unless_native
  - with_native

Unless type hints are provided, these columns will not be materialized in the destination.
One way to provide type hints is to use the 'columns' argument in the '@dlt.resource' decorator.  For example:

@dlt.resource(columns={'capability_id': {'data_type': 'text'}})

Pipeline logfire_ingest load step completed in 0.21 seconds
1 load package(s) were loaded to destination duckdb and into dataset agent_traces
The duckdb destination used duckdb:////Users/niteshmishra/LLM_Zoomcamp_new/LLM_Zoomcamp/06_dlt_workshop/.dlt/data/dev/logfire_ingest.duckdb location to store data
Load package 1785194247.816145 is LOADED and contains no failed jobs
```

4. Verify with:
```sql
SELECT COUNT(*) FROM information_schema.tables
WHERE table_schema = 'agent_traces';

uv run python -c "
import duckdb
con = duckdb.connect('.dlt/data/dev/logfire_ingest.duckdb')
print(con.sql(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'agent_traces'\").fetchall())
"
```

Output
```bash
06_dlt_workshop % uv run python -c "
import duckdb
con = duckdb.connect('.dlt/data/dev/logfire_ingest.duckdb')
print(con.sql(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'agent_traces'\").fetchall())
"
[(22,)]
```

Answer - ~24

Extra info 
```bash
 06_dlt_workshop % uv run python -c "
import duckdb
con = duckdb.connect('.dlt/data/dev/logfire_ingest.duckdb')
print(con.sql(\"SELECT * FROM information_schema.tables WHERE table_schema = 'agent_traces'\").fetchall())
"
[('logfire_ingest', 'agent_traces', 'spans', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__gen_ai_input_messages', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__gen_ai_input_messages__parts', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__gen_ai_input_messages__parts__result', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__gen_ai_output_messages', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__gen_ai_output_messages__parts', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__gen_ai_response_finish_reasons', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__gen_ai_system_instructions', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__gen_ai_tool_call_result', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__gen_ai_tool_definitions', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__gen_ai_tool_definitions__parameters__required', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__logfire_metrics__gen_ai_client_token_usage__details', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__logfire_metrics__operation_cost__details', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__model_request_parameters__function_tools', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__model_request_parameters__function_tools__parameters_json_schema__required', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__model_request_parameters__instruction_parts', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__pydantic_ai_all_messages', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__pydantic_ai_all_messages__parts', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', 'spans__attributes__pydantic_ai_all_messages__parts__result', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', '_dlt_loads', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', '_dlt_pipeline_state', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None), ('logfire_ingest', 'agent_traces', '_dlt_version', 'BASE TABLE', None, None, None, None, None, 'YES', 'NO', None, None)]
```

# Question 3. Query traces with an agent
Using a coding agent (you can also write the code by hand) find the input token usage for the agent run from Q1.

The token counts are stored in the span attributes as gen_ai.usage.input_tokens. Sum them across all LLM calls within the trace. The number depends on how many searches the agent made, so report the range it falls into:

- 100 - 500
- 1500 - 5000 <- answer
- 10000 - 20000
- 50000 - 100000

## Steps followed 

Step 1 — confirm the token column names on spans:

```bash
uv run python -c "
import duckdb
con = duckdb.connect('.dlt/data/dev/logfire_ingest.duckdb')
for r in con.sql(\"SELECT column_name FROM information_schema.columns WHERE table_schema='agent_traces' AND table_name='spans' AND (column_name ILIKE '%token%' OR column_name ILIKE '%usage%')\").fetchall():
    print(r[0])
"

# Output
attributes__gen_ai_aggregated_usage_input_tokens
attributes__gen_ai_aggregated_usage_output_tokens
attributes__logfire_metrics__gen_ai_client_token_usage__total
attributes__gen_ai_usage_input_tokens
attributes__gen_ai_usage_output_tokens
```

Step 2 — find the trace_id for your Q1 "Ollama" run:

```bash
uv run python -c "
import duckdb
con = duckdb.connect('.dlt/data/dev/logfire_ingest.duckdb')
print(con.sql(\"SELECT trace_id, span_name, message FROM agent_traces.spans WHERE attributes__final_result ILIKE '%Ollama%'\").fetchall())
"

# Output
[('019fa5c7f0e32d2bb66bbf907da71d6d', 'invoke_agent faq_agent', 'faq_agent run')]
```

Step 3 — once you have both, sum input tokens across that trace's LLM-call spans (swap in the real column name from Step 1 if it differs from my guess, and paste in the real trace_id from Step 2):

```bash
uv run python -c "
import duckdb
con = duckdb.connect('.dlt/data/dev/logfire_ingest.duckdb')
print(con.sql(\"\"\"
    SELECT SUM(attributes__gen_ai_usage_input_tokens) AS total_input_tokens,
           COUNT(*) AS n_llm_calls
    FROM agent_traces.spans
    WHERE trace_id = '019fa5c7f0e32d2bb66bbf907da71d6d'
      AND attributes__gen_ai_usage_input_tokens IS NOT NULL
\"\"\").fetchall())
"

# Output
[(1487, 2)]
```

1487 total input tokens across 2 LLM calls. So your Q1 "Ollama" run only needed one search round (matches the earlier span count you found: root + 2 chat calls + 1 tool call = 4 spans, consistent with your span-count answer of 5).

That 1487 sits just under the 1500 boundary, but these options are order-of-magnitude buckets, not tight ranges — 1487 is unambiguously in the thousands, not the hundreds (100-500 would need this to be 3x smaller). So 1500-5000 is your answer for Q3, with the understanding that "1487 ≈ ~1500" given the coarse bucketing.


Step 4 — cross-check against the root span's own pre-aggregated total (should roughly match Step 3's sum):
```bash
uv run python -c "
import duckdb
con = duckdb.connect('.dlt/data/dev/logfire_ingest.duckdb')
print(con.sql(\"\"\"
    SELECT attributes__gen_ai_aggregated_usage_input_tokens
    FROM agent_traces.spans
    WHERE trace_id = '019fa5c7f0e32d2bb66bbf907da71d6d'
      AND message = 'faq_agent run'
\"\"\").fetchall())
"

# Output
[(1487,)]
```

Answer - 1500 - 5000