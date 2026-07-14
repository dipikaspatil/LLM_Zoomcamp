# Homework: Evaluation

In homework 2 we built keyword, vector, and hybrid search over the course lessons, and ended with an open question: which one is best? The way to answer that is to measure, and that's what we do here.

In this homework we generate a ground truth dataset and use it to evaluate search, the same way we did in the module. There we only evaluated keyword search. Here we also evaluate vector and hybrid search, so we can finally compare them on numbers instead of intuition.

Like in homework 1 and 2, our knowledge base is the course lessons themselves. Each module has a lessons/ folder of numbered markdown pages, and we pull them from GitHub. We use commit 8c1834d, so everyone works with the exact same 72 pages.

It's possible your answers won't match exactly. If so, select the closest one.


# Setup

This homework continues from homework 2. We reuse the same chunks and the same search functions, so it's easiest to keep working in the same project.

We need a few more libraries for generating questions with an LLM:

`uv add openai pydantic python-dotenv pandas gitsource minsearch`

For the LLM, we recommend OpenAI with gpt-5.4-mini, but you can use any model and provider you like - just adapt the client accordingly. Put your key in a .env file as in the earlier modules.

Load the data exactly as in homework 2:

```python
# Loading the docs
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]
```

This gives 72 pages.

# Generating ground truth

To evaluate search, we need a dataset of questions where we know which document is the correct answer. This is the ground truth.

We generate it the same way as in the module. For each lesson page, we ask an LLM to write 5 questions that are answered by that page. Each question is then labeled with the page it came from.

We use the same structured-output approach as in the module - the same Questions model and the llm_structured helper from evaluation_utils.py.

Download evaluation_utils.py and the rag_helper.py it depends on:

```bash
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main
wget ${PREFIX}/01-agentic-rag/code/rag_helper.py
wget ${PREFIX}/04-evaluation/code/evaluation_utils.py
```

The module's instructions generate questions from a FAQ record, so we adapt them for a lesson page:

```python
data_gen_instructions = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()
``` 

We ask for different wording from the page on purpose. Real users don't phrase their questions the way the lesson does, and copying the text would make the evaluation too easy.

For each page, build a JSON user prompt from its filename and content, then call llm_structured with the Questions model. Turn each returned question into a record labeled with the page's filename. The call also returns the token usage, the same as in the lessons.

# Q1. Generating questions

Generating questions for all 72 pages costs money and takes time, so let's start small and generate questions for just the first 3 pages:

- 01-agentic-rag/lessons/01-intro.md
- 01-agentic-rag/lessons/02-environment.md
- 01-agentic-rag/lessons/03-rag.md

Each call returns the token usage, which most LLM APIs report on the response object (e.g. response.usage.input_tokens / prompt_tokens).

What's the average number of input tokens across these 3 calls?

- 140
- 1400 <-- answer
- 14000
- 140000

These numbers vary between runs, even with the same model, so pick the closest option. A different provider or model may land further apart, but the input tokens stay in the same order of magnitude - the prompt we send is the same.

## What's happening conceptually: 
same idea as Lesson 02/03 (generate ~5 synthetic questions per document via structured LLM output), but now you're asked to inspect token usage, not just the questions themselves — this is the "cost tracking" habit from the lessons, made explicit.

## How to do it

```python
results = []
for doc in first_three_pages:          # the 3 specified lesson files
    response = llm_structured(doc, Questions)   # however evaluation_utils exposes this
    results.append(response.usage.input_tokens)  # or .prompt_tokens, check the actual response object

average_input_tokens = sum(results) / len(results)
print(average_input_tokens)
```
## How to sanity-check your own number before picking an option: 
each chunk/doc here can be up to ~2000 characters of markdown, and roughly 4 characters ≈ 1 token in English text. So a single ~2000-character doc is roughly 500 tokens of raw content, plus your prompt template (instructions, field descriptions, JSON schema for structured output) adds a few hundred more. That puts you solidly in "low thousands," not tens or hundreds of thousands — so before trusting whatever number your code prints, ask "is it the right order of magnitude given a ~2000-char input?" This is the same estimation instinct as Lesson 02's cost-tracking.

## Output
```shell
01-agentic-rag/lessons/01-intro.md -> 908 input tokens
01-agentic-rag/lessons/02-environment.md -> 1122 input tokens
01-agentic-rag/lessons/03-rag.md -> 1534 input tokens

Average input tokens: 1188.0
```
Average is 1188 input tokens per call, which lines up almost exactly with the order-of-magnitude estimate from earlier (~500 tokens raw content + prompt overhead, scaled up a bit since these lesson pages ran longer than a bare 2000-char chunk).

Of the four options — 140 / 1400 / 14000 / 140000 — 1400 is the closest match to 1188.

----------------------------------------------------------------------------------------------------------

# The full ground truth

You don't need to generate the data for the rest of the homework. We already did it for all 72 pages, using the same approach as in the lessons, and saved the 360 questions to a file.

```bash
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main
wget ${PREFIX}/cohorts/2026/04-evaluation/ground-truth.csv
```
Load it with pandas into a list of records called ground_truth. Each record has a question and the filename of the page that should answer it.

# Searching the chunks
We search over the same chunks as in homework 2.

Create them with chunk_documents:

```python
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
```

This gives 295 chunks.

Now rebuild the search from homework 2 over these chunks. Build a text index (Index) and a vector index (VectorSearch), both keyed on filename. Wrap each one in a function, text_search and vector_search, that takes a query and the number of results to return (5 by default).

For hybrid search, reuse the rrf function from homework 2:

```python
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]
```

Then define hybrid_search on top of it:

```python
def hybrid_search(query, k=60):
    text_results = text_search(query, num_results=10)
    vector_results = vector_search(query, num_results=10)
    return rrf([text_results, vector_results], k=k)
```

# Q2. First result with text search
Take the first question from the ground truth:

`q = ground_truth[0]["question"]`

After running text_search for it, what's the filename of the first result?

- 01-agentic-rag/lessons/01-intro.md
- 01-agentic-rag/lessons/03-rag.md <-- answer
- 01-agentic-rag/lessons/13-function-calling.md
- 01-agentic-rag/lessons/10-rag-next-steps.md

# Q3. First result with vector search
After running vector_search for the same question, what's the filename of the first result?

- 01-agentic-rag/lessons/01-intro.md <-- answer
- 01-agentic-rag/lessons/03-rag.md
- 04-evaluation/lessons/11-evaluation-intro.md
- 04-evaluation/lessons/12-rag-answers.md

This question was generated from 01-agentic-rag/lessons/01-intro.md. Notice that one method finds the right page at the top and the other doesn't. That's exactly why we measure across the whole dataset instead of trusting one query.

## Q2 & Q3 — First result from text search vs. vector search

Concept: you're directly comparing keyword/BM25-style search (`text_search`, built on an inverted index over chunk text) against embedding-based search (`vector_search`, built on a `VectorSearch` index over chunk embeddings) — same first question, different retrieval mechanism, see if they even agree on the top hit.

```python
ground_truth = pd.read_csv("ground-truth.csv")
first_question = ground_truth.iloc[0]["question"]

text_results = text_search(first_question, num_results=5)
print(text_results[0]["filename"])

vector_results = vector_search(first_question, num_results=5)
print(vector_results[0]["filename"])
```

## 1. Copy the two small embedder files into 04_evaluation/:

```bash
cp /Users/niteshmishra/LLM_Zoomcamp_new/LLM_Zoomcamp/02_Vector_Search/embedder.py .
cp /Users/niteshmishra/LLM_Zoomcamp_new/LLM_Zoomcamp/02_Vector_Search/download.py .
```

## 2. Add the ONNX runtime dependencies 

`uv add onnxruntime tokenizers huggingface_hub`

## 3. Download the model once (same as we did in Module 02):

`python3 download.py`

This creates models/Xenova/all-MiniLM-L6-v2/{tokenizer.json, model.onnx} inside 04_evaluation/.

## Output
```bash
04_evaluation % python3 q2_q3_first_search.py
Embedding 295 chunks...

Question: What exactly is a retrieval-augmented generation system, and why does it help with answers that the model wouldn't know on its own?
Q2 - text_search top result: 01-agentic-rag/lessons/03-rag.md
Q3 - vector_search top result: 01-agentic-rag/lessons/01-intro.md
```
----------------------------------------------------------------------------------------------------------------------------

# Evaluation metrics
We evaluate search exactly as in the module, reusing the same functions from the lecture. We change only the label. Our ground truth uses filename, so a result counts as a hit when a returned chunk's filename matches the question's filename, not a document id.

As a reminder, these functions do the following:

- compute_relevance runs search for a question and returns a list of 0s and 1s
- hit_rate is the fraction of questions where the correct page appears in the results
- mrr (Mean Reciprocal Rank) also rewards finding the page near the top
- evaluate runs a search function over the whole ground truth and returns both metrics

----------------------------------------------------------------------------------------------------------------------------
# Q4. Evaluating text search
Evaluate text_search on the ground truth data.

What's the Hit Rate?

- 0.55
- 0.66
- 0.76 <-- answer
- 0.88

## Hit Rate for text_search across all 360 questions

This is exactly the evaluate() + hit_rate() pattern from Lesson 04/05, just applied with filename as the relevance key instead of document:

## Output 
```bash
04_evaluation % python3 q4_evaluate_text_search.py
Hit Rate for text_search across all 360 questions: 0.7583333333333333
```

----------------------------------------------------------------------------------------------------------------------------

# Q5. Evaluating vector search
Now evaluate vector_search - the part we left for the homework, since the module only evaluated keyword search.

What's the MRR?

- 0.35
- 0.45
- 0.55 <-- answer
- 0.65

## Output
```bash
04_evaluation % python3 q5_evaluate_vector_search.py
Embedding 295 chunks...
MRR for vector_search across all 360 questions: 0.5486111111111112
```

----------------------------------------------------------------------------------------------------------------------------

# Q6. Tuning hybrid search
The k constant in RRF controls how much the top ranks matter. A smaller k sharpens the gap between positions, so being at the top of a list counts for more. The RRF paper uses 60 as a default, but the best value depends on the data

so let's measure it.
Evaluate hybrid_search over the full ground truth dataset for k values 1, 50, 100, and 200. Compare the MRR values for these runs.

Which k gives the best MRR?

- 1 <-- answer
- 50
- 100
- 200

Several values of k may give the same MRR. If there's a tie, pick the smallest k.

## Hybrid search: tuning the RRF parameter k

`Reciprocal Rank Fusion (RRF)` combines two ranked lists (e.g., text search results and vector search results) into one ranked list, without needing the two systems' scores to be on comparable scales (BM25 scores and cosine similarities aren't directly comparable numbers). Instead, RRF only looks at rank position:

```python
RRF_score(doc) = Σ  1 / (k + rank_in_list)
                over every list the doc appears in
```

- A document ranked #1 in a list contributes 1/(k+1); ranked #5 contributes 1/(k+5).
- Sum contributions across all lists the document appears in (a doc found near the top of both text and vector search gets a much bigger combined score than one found in only one list).
- Sort all documents by total RRF score, descending — that's your fused ranking.

What k controls: it's a smoothing/damping constant.
- `Small k` (e.g. 1) → the formula is very sensitive to exact rank; being #1 vs #2 makes a huge relative difference (1/2 vs 1/3 — a 33% drop).
- `Large k` (e.g. 200) → 1/(200+1) vs 1/(200+2) are nearly identical, so rank position barely matters anymore, and the fusion behaves more like "which docs simply appeared in both lists" rather than "which docs ranked highest."

## Output
```bash
04_evaluation % python3 q6_tuning_hybrid_search.py
Embedding 295 chunks...
Hybrid search: tuning the RRF parameter k: 1 , mrr: 0.6513425925925931
Hybrid search: tuning the RRF parameter k: 50 , mrr: 0.5926851851851856
Hybrid search: tuning the RRF parameter k: 100 , mrr: 0.5926851851851856
Hybrid search: tuning the RRF parameter k: 200 , mrr: 0.5926851851851856
```