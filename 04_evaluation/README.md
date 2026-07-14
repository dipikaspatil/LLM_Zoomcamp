# Module 4: Evaluation — Full Lesson Notes

Deep-dive notes on every lesson in [DataTalksClub/llm-zoomcamp — 04-evaluation](https://github.com/DataTalksClub/llm-zoomcamp/tree/main/04-evaluation).
Goal: systematically evaluate **search**, **RAG**, and **agent** systems instead of eyeballing results.

---

## Table of Contents

**Part 1 — Search Evaluation**
1. [Intro to Evaluation](#01-intro-to-evaluation)
2. [Ground Truth Generation](#02-ground-truth-generation)
3. [Ground Truth at Scale (Batch)](#03-ground-truth-at-scale-batch)
4. [Search Evaluation](#04-search-evaluation)
5. [Search Metrics — Hit Rate & MRR](#05-search-metrics--hit-rate--mrr)
6. [Search Parameter Tuning](#06-search-parameter-tuning)

**Part 2 — RAG & Agent Evaluation**
7. [RAG/Agent Evaluation Intro](#11-ragagent-evaluation-intro)
8. [Generating RAG Answers](#12-generating-rag-answers)
9. [LLM-as-a-Judge](#13-llm-as-a-judge)
10. [Agent Evaluation](#14-agent-evaluation)
11. [Next Steps](#15-next-steps)

---

## 01. Intro to Evaluation

**Why this matters:** Without a systematic way to compare approaches, you're stuck guessing whether keyword search, vector search, a new prompt, or a different model is actually better. Evaluation replaces "it feels better" with a number.

**The core trick — A → Q\* → A′:**
Since you rarely have real user queries to start with, you generate them synthetically from your own data:

```
A  = an existing FAQ answer (ground truth document)
Q* = an LLM-generated question that this answer would answer
A' = what your system produces when given Q*
```

You then check: did search retrieve the original document (A) for Q\*? Did RAG produce an answer close to the original?

**Offline vs. online evaluation:**

| Type | When | Purpose |
|---|---|---|
| Offline | Before deploy, on a fixed dataset | Compare methods/parameters/models |
| Online | After deploy, on real traffic | Monitor real-world quality, catch drift |

**Three levels of evaluation, increasing in scope:**
1. **Search** — did we retrieve the right document?
2. **RAG** — given the right (or wrong) document, did we generate the right answer?
3. **Agent** — did the agent call the right tools, in a sensible order, and land on a good final answer?

> Key principle: fix retrieval first. If search brings back the wrong documents, no prompt engineering or bigger model will save the final answer.

**Caveat:** synthetic questions tend to closely mirror the source document's vocabulary, which can inflate metrics. Real user queries are messier and should eventually replace/augment synthetic ones.

---

## 02. Ground Truth Generation

**Goal:** produce labeled `(question, document_id)` pairs — a dataset where we *know* the correct answer to each question ahead of time.

**Approach:** for each FAQ document, ask an LLM to generate ~5 plausible student questions that the document answers — *without* copying its exact wording (otherwise search would trivially match on shared vocabulary).

**Structured output with Pydantic** — instead of parsing free-form text, force the LLM to return a typed object:

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class Questions(BaseModel):
    questions: list[str]

def generate_questions(doc: dict) -> Questions:
    prompt = f"""
You are a student in a data engineering/ML course.
Formulate 5 questions this student might ask based on the FAQ record below.
The questions should be complete and not too short.
Use as few words from the record as possible; write them in your own words.

section: {doc['section']}
question: {doc['question']}
answer: {doc['text']}
""".strip()

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=prompt,
        text_format=Questions,
    )
    return response.output_parsed
```

**Building the ground truth records:**

```python
ground_truth = []

for doc in documents:            # documents filtered to the course you care about
    result = generate_questions(doc)
    for q in result.questions:
        ground_truth.append({"question": q, "document": doc["id"]})
```

Each row says: "if a user asks this question, `document` is the correct/expected result."

**Cost tracking matters** — every LLM call has a token cost; the lesson wraps calls with helpers that sum prompt/completion tokens so you know what a full run costs before scaling up.

---

## 03. Ground Truth at Scale (Batch)

**Goal:** run the single-document generator from Lesson 02 across the *entire* document set, reliably and quickly.

**Reliability — retries:** API calls fail transiently (timeouts, rate limits). Wrap the call so one bad request doesn't kill an hour-long batch job:

```python
import time

def llm_structured_retry(fn, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)   # simple exponential backoff
```

**Speed — parallelize with threads:** LLM calls are I/O-bound (mostly waiting on the network), so a thread pool gives a big speedup over sequential calls:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

def generate_ground_truth(doc):
    questions = llm_structured_retry(generate_questions, doc)
    return [{"question": q, "document": doc["id"]} for q in questions.questions]

def map_progress(fn, items, max_workers=6):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())
    return results

records_per_doc = map_progress(generate_ground_truth, documents, max_workers=6)
ground_truth = [row for records in records_per_doc for row in records]
```

`max_workers=6` is a conservative default to avoid tripping API rate limits.

**Persisting the dataset:**

```python
import pandas as pd

df_ground_truth = pd.DataFrame(ground_truth)
df_ground_truth.to_csv("ground-truth-data.csv", index=False)
```

For reference scale: ~79 documents → ~395 generated questions → roughly $0.06 in API cost.

---

## 04. Search Evaluation

**Goal:** for every ground-truth question, run it through search and check whether the expected document comes back.

**A generic, swappable search function** — this is the key design decision: write evaluation code against a `search_function(query)` interface so you can later drop in vector or hybrid search without touching the evaluation logic.

```python
def text_search(query, index, num_results=5):
    boost = {"question": 3.0, "section": 0.5}
    return index.search(
        query=query,
        boost_dict=boost,
        num_results=num_results,
    )
```

**Turning results into relevance labels** — for each query, mark each of the top-k results `1` if it matches the expected document, `0` otherwise:

```python
def compute_relevance(query_result, expected_doc_id):
    return [1 if doc["id"] == expected_doc_id else 0 for doc in query_result]

def evaluate(ground_truth, search_function):
    relevance_total = []
    for record in ground_truth:
        results = search_function(record["question"])
        relevance = compute_relevance(results, record["document"])
        relevance_total.append(relevance)
    return relevance_total
```

Example output for one query with 5 results, correct doc ranked first: `[1, 0, 0, 0, 0]`.
This list of relevance lists is the raw material for the metrics in Lesson 05.

---

## 05. Search Metrics — Hit Rate & MRR

**Hit Rate (a.k.a. Recall@k):** fraction of queries where the correct document appears *anywhere* in the top-k results. Ignores rank/position.

```python
def hit_rate(relevance_total):
    hits = sum(1 for line in relevance_total if 1 in line)
    return hits / len(relevance_total)
```

**Mean Reciprocal Rank (MRR):** rewards ranking the correct document *higher*. For each query, the score is `1 / rank` of the correct document (rank starts at 1); `0` if it's missing entirely.

```python
def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank, is_relevant in enumerate(line, start=1):
            if is_relevant:
                total_score += 1 / rank
                break
    return total_score / len(relevance_total)
```

**Combined evaluation:**

```python
def evaluate_metrics(ground_truth, search_function):
    relevance_total = evaluate(ground_truth, search_function)
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }
```

**Worked example (15 queries):** Hit Rate = 0.933 (14/15 had the correct doc somewhere in top-k) but MRR = 0.822 (lower, because not every hit was ranked #1).

⚠️ Because ground truth questions are LLM-generated from the documents themselves, scores above ~95% can be artificially inflated — vocabulary overlap makes retrieval "too easy" compared to messy real-world queries. Treat these numbers as a *relative* comparison tool, not an absolute quality score.

---

## 06. Search Parameter Tuning

**Goal:** use Hit Rate/MRR from the fixed ground-truth set to tune search parameters empirically instead of by intuition — and the lesson's biggest lesson is that intuition is often wrong.

**Counter-intuitive finding:** boosting the `question` field heavily (e.g. 3.0×) — which *sounds* like it should help, since ground-truth questions resemble the `question` field — actually **hurt** MRR/Hit Rate compared to no boost at all (boost = 1.0 scored best: Hit Rate 0.924, MRR 0.814).

**Single-parameter sweep:**

```python
def evaluate_boost(ground_truth, index, question_boost):
    def search_fn(query):
        return index.search(
            query=query,
            boost_dict={"question": question_boost},
            num_results=5,
        )
    return evaluate_metrics(ground_truth, search_fn)

for qb in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    print(qb, evaluate_boost(ground_truth, index, qb))
```

**Grid search over multiple parameters:**

```python
import itertools

param_grid = {
    "question": [1.0, 2.0, 5.0],
    "answer":   [1.0, 2.0, 4.0, 10.0],
    "section":  [0.1, 0.2, 0.5],
}

best = None
for question, answer, section in itertools.product(*param_grid.values()):
    boost = {"question": question, "answer": answer, "section": section}
    def search_fn(query, boost=boost):
        return index.search(query=query, boost_dict=boost, num_results=5)

    metrics = evaluate_metrics(ground_truth, search_fn)
    if best is None or metrics["mrr"] > best["mrr"]:
        best = {**metrics, "boost": boost}

print(best)
```

**Winning configuration:** `question=1.0, answer=2.0, section=0.1` — i.e., weight the **answer** field twice as heavily as the question, and give the section field almost no weight. This contradicted the initial hypothesis.

**Practical notes:**
- Grid search is fine for a handful of parameters; it explodes combinatorially beyond that.
- For larger search spaces, use Bayesian optimization (e.g. `hyperopt`) instead of brute force.
- `num_results` (top-k) is itself a tradeoff: higher k → higher Hit Rate, but more context tokens (cost/latency) fed to the LLM downstream in RAG.

---

## 11. RAG/Agent Evaluation Intro

**Why go beyond search evaluation:** once retrieval is "good enough," the next failure mode is generation — the LLM might get the right context and still produce a bad answer (or the reverse: bad context but a lucky good answer). You can't just string-match `A` against `A′` because paraphrasing is expected and desirable.

**Extending the framework to full pipelines:**
- **RAG:** Question → retrieve context → generate answer → compare generated answer to the original.
- **Agent:** Question → agent decides tool calls → produces answer → evaluate *both* the final answer and the sequence of tool calls (the "trajectory").

**LLM-as-a-judge, at a glance:** feed a judge model three things — the question, the reference (original) answer, and the generated answer — and ask it to decide if they're semantically equivalent.

> Asking the judge to explain *why* it made a decision (not just give a verdict) tends to produce more reliable classifications, and gives you something to read when debugging failures.

When an answer fails, the root cause could be: retrieval brought back the wrong document, the prompt was poorly structured, or the LLM ignored/misused good context — the judge's reasoning helps you tell these apart.

---

## 12. Generating RAG Answers

**Setup — the A → Q → A′ pipeline in full:**
- `A` = original FAQ answer (ground truth)
- `Q` = ground-truth question (from Lesson 02/03)
- `A'` = answer generated by running `Q` through your actual RAG system

```python
doc_idx = {d["id"]: d for d in documents}   # id -> full document, for looking up A

def generate_rag_answer(rec):
    question = rec["question"]
    answer_llm = assistant.rag(question)          # your RAG pipeline: search + prompt + LLM call
    answer_orig = doc_idx[rec["document"]]["text"]

    return {
        "question": question,
        "answer_llm": answer_llm,
        "answer_orig": answer_orig,
        "document": rec["document"],
    }
```

**Running it over the full ground-truth set, in parallel:**

```python
results = map_progress(generate_rag_answer, ground_truth, max_workers=6)

df_results = pd.DataFrame(results)
df_results.to_csv("rag-evaluation-data.csv", index=False)
```

A token-usage wrapper (`RAGWithUsage` in the original material) tracks cost alongside each call. Reference scale: 395 questions → ~$0.34 to generate all RAG answers. This CSV — `question / answer_llm / answer_orig / document` — is exactly the input the judge in Lesson 13 needs.

---

## 13. LLM-as-a-Judge

**Goal:** automatically score whether `answer_llm` (RAG output) is a *semantically* acceptable substitute for `answer_orig` (ground truth) — without requiring exact wording.

**Structured judge output:**

```python
from pydantic import BaseModel
from typing import Literal

class AnswerEvaluation(BaseModel):
    reasoning: str
    score: Literal["good", "bad"]
```

**Judge prompt — the key instruction is "semantically equivalent," not "identical":**

```python
def evaluate_aqa(question, answer_orig, answer_llm):
    prompt = f"""
You are comparing an AI-generated answer to a reference answer for the same question.
Decide whether the AI answer is semantically equivalent to the reference answer.
The AI answer does NOT need to be word-for-word identical — different phrasing,
extra helpful detail, or a different structure is fine as long as the key
information is correct and nothing important is missing or wrong.

Question: {question}
Reference answer: {answer_orig}
AI answer: {answer_llm}

Give your reasoning, then a final score of "good" or "bad".
""".strip()

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=prompt,
        text_format=AnswerEvaluation,
    )
    return response.output_parsed
```

**Running the judge over every row, in parallel** (same `map_progress` pattern as Lessons 03/12):

```python
def judge_row(rec):
    result = evaluate_aqa(rec["question"], rec["answer_orig"], rec["answer_llm"])
    return {**rec, "score": result.score, "reasoning": result.reasoning}

judged = map_progress(judge_row, results, max_workers=6)
df_judged = pd.DataFrame(judged)
df_judged["score"].value_counts()
```

**Reference result:** 379 good / 16 bad out of 395 (~96%), for about $0.25 in judge-call cost.

**The judge is not the final word — read the failures.** The lesson stresses building a small review UI (e.g. Streamlit) to look at question / reference / generated answer / judge reasoning side-by-side for the "bad" cases, so you can tell whether the *real* culprit is retrieval, prompt design, or the judge itself being wrong — and refine the judge's instructions based on what you find.

---

## 14. Agent Evaluation

**What's different from RAG evaluation:** an agent doesn't just answer — it *decides* what tool(s) to call and in what order. A bad final answer might stem from a bad **decision to search for the wrong thing**, not a reasoning failure. So you evaluate two things: the final **answer** and the **trajectory** (sequence of tool calls) that produced it.

**Capturing the trajectory:** run the agent (e.g. with a toolkit like ToyAIKit wrapping an OpenAI model + a `search` tool) and record every tool call it makes alongside the final answer:

```python
def run_agent(rec):
    question = rec["question"]
    agent_response = agent.run(question)   # e.g. agent with a `search` tool available

    tool_calls = [
        {"tool": call.name, "arguments": call.arguments}
        for call in agent_response.tool_calls
    ]

    return {
        "question": question,
        "answer_llm": agent_response.final_answer,
        "answer_orig": doc_idx[rec["document"]]["text"],
        "tool_calls": tool_calls,
    }
```

Example captured call: agent searches `"own pace certificate at the end self-paced course certificate"` for a question about self-paced certificates — you can literally read what the agent decided to look for.

**Judging both dimensions with one structured model:**

```python
class AgentEvaluation(BaseModel):
    answer_reasoning: str
    answer_score: Literal["good", "bad"]
    trajectory_reasoning: str
    trajectory_score: Literal["good", "bad"]
```

The trajectory judge prompt asks it to check:
- Were the search queries relevant to the question?
- Did they include the important keywords?
- Were there duplicate/redundant searches?
- If multiple searches happened, did later ones meaningfully refine earlier ones?
- Was the number of calls reasonable? (1 call = ideal, 2–3 = acceptable, 3+ needs justification in the reasoning)

**Reference results (50 agent runs, ~$0.05 to judge):**

| Dimension | Good | Bad |
|---|---|---|
| Answer correctness | 45 | 5 |
| Trajectory quality | 49 | 1 |

Trajectory quality being higher than answer quality here suggests the agent was mostly searching sensibly — the few wrong final answers weren't primarily a tool-use problem.

---

## 15. Next Steps

**Recap of the three evaluation levels built across this module:**
1. Search → Hit Rate, MRR
2. RAG → LLM-as-a-judge on generated answers
3. Agents → LLM-as-a-judge on both answers and tool-call trajectories

**Evaluation is not a one-time step.** Every time you change a prompt, swap a model, adjust a retrieval parameter, or add a tool, re-run the evaluation suite before shipping — regressions are easy to introduce invisibly otherwise.

**Path from synthetic to real evaluation data:**
1. Start with synthetic ground truth generated from your docs (as in Lessons 02–03).
2. If metrics look suspiciously high, revisit/tighten the question-generation prompt.
3. Ship, and start collecting real user queries.
4. Get human labels on a sample of real queries.
5. Use what you learn from real queries to improve the *synthetic* generator, closing the loop.

**Manual evaluation still matters.** No automated metric replaces personally using the system, hunting for edge cases, and writing down concrete failure examples — this is where judge-prompt refinements and product decisions actually come from.

**Frameworks worth knowing for deeper/production evaluation:**

| Tool | Focus |
|---|---|
| [Ragas](https://github.com/explodinggradients/ragas) | RAG-specific metrics: faithfulness, context precision/recall |
| [DeepEval](https://github.com/confident-ai/deepeval) | Hallucination detection, unit-test-style LLM evaluation |
| [TruLens](https://github.com/truera/trulens) | Tracking quality metrics over time, feedback functions |

**For post-deployment (online) monitoring:** collect explicit user feedback (thumbs up/down), log full request/response traces, build metric dashboards, and set up alerting for quality degradation over time.

---

## Supporting Materials in the Repo

- `code/` — Jupyter notebooks implementing each lesson end-to-end
- `data/` — FAQ/document datasets used for ground truth and evaluation
- `README.md` — original module overview
- Homework: `cohorts/2026/04-evaluation/`
- Full workshop recording linked from the module README

## Glossary

| Term | Meaning |
|---|---|
| Ground truth | A dataset of (query, correct-document) pairs used as the answer key for evaluation |
| Hit Rate / Recall@k | % of queries where the correct doc appears anywhere in the top-k results |
| MRR | Average of `1/rank` of the correct doc across queries — rewards ranking it higher |
| LLM-as-a-judge | Using an LLM to grade another LLM's output against a reference answer |
| Trajectory | The sequence of tool calls an agent makes before producing a final answer |
| Offline evaluation | Evaluation on a fixed, known dataset (used throughout this module) |
| Online evaluation | Evaluation on live production traffic and real user feedback |