# Module 2: Vector Search

This module extends the RAG pipeline from Module 1 with **vector search** —
matching documents by semantic meaning instead of exact keyword overlap. It
goes from raw embeddings to persistent vector indexes (sqlitesearch,
PGVector) and ONNX-based embedders for lightweight deployments.

- Code: [code/](code/)
- Embeddings runtime: [embed/](embed/)
- Full workshop recording: [Vector Databases: Embeddings, Semantic Search, and Hybrid Retrieval](https://www.youtube.com/watch?v=BC3NsRUNEIg)
- Homework: [HOMEWORK.md](HOMEWORK.md)

---

## Table of Contents

1. [What is Vector Search](#1-what-is-vector-search)
2. [Embeddings](#2-embeddings)
3. [Embedding Our Dataset](#3-embedding-our-dataset)
4. [Vector Search (with numpy)](#4-vector-search-with-numpy)
5. [Vector Search with minsearch](#5-vector-search-with-minsearch)
6. [RAG with Vector Search](#6-rag-with-vector-search)
7. [Vector Search with sqlitesearch](#7-vector-search-with-sqlitesearch)
8. [Vector Search with PGVector](#8-vector-search-with-pgvector)
9. [ONNX Embedder](#9-onnx-embedder-optional)
10. [Next Steps](#10-next-steps)

---

## 1. What is Vector Search

Keyword search (module 1) only matches exact words — "Can I still join the
course after the start date?" and "Is it possible to enroll late?" mean the
same thing but share almost no words. **Vector search matches meaning, not
words.**

**Two-stage process:**
1. **Offline (indexing):** convert all documents into vectors and store them in an index.
2. **Online (querying):** convert the user's query into a vector with the *same* model, then find the closest document vectors.

An embedding model is a neural net trained so that similar meanings land on
similar vectors. Similarity is usually measured with **cosine similarity**
(the angle between two vectors):

| Relationship | Cosine similarity |
|---|---|
| Same direction (similar) | ≈ 1 |
| Right angles (unrelated) | ≈ 0 |
| Opposite direction (opposite meaning) | ≈ -1 |

**Keyword vs. vector search:**

| | Keyword search | Vector search |
|---|---|---|
| Matches | Exact words | Meaning |
| Best for | Specific terms, IDs, names | Paraphrased / natural language |
| Index | Inverted index (BM25, TF-IDF) | Vector index (cosine similarity) |
| Weakness | Misses synonyms/paraphrases | Misses exact term matches |

> **Advice:** Never start with vector search. Start with text search, and
> reach for vectors only once you can show they're worth the extra
> operational cost. In practice the two work best combined — see *Hybrid
> Search* in the Best Practices module.

This module builds vector search three ways and plugs each into RAG:
1. **minsearch** — in-memory, simplest, good for experiments
2. **sqlitesearch** — persistent, SQLite-backed, production-friendly
3. **PGVector** — Postgres + pgvector, scalable, Docker-based

**Setup** (separate project recommended — `sentence-transformers` pulls in PyTorch and is heavy):
```bash
mkdir llm-zoomcamp-code && cd llm-zoomcamp-code
uv init
uv add requests minsearch openai jupyter python-dotenv
```

---

## 2. Embeddings

Embedding = turning text into a fixed-length vector such that similar
meanings produce similar vectors. Rooted in [word2vec](https://en.wikipedia.org/wiki/Word2vec)
(word-level), extended to whole sentences (sentence embeddings) — the model
encodes context, so the same word ("judge") gets different vectors in a
legal sentence vs. an ML-evaluation sentence.

We use [sentence-transformers](https://www.sbert.net/) (local, no API cost):
```bash
uv add sentence-transformers
```

**Model choice:** `all-MiniLM-L6-v2` — 384-dim, fast on CPU, good general
English quality, outputs normalized (unit-length) vectors.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
```

**Quick similarity check:**
```python
q1 = "Can I still join the course after the start date?"
v1 = model.encode(q1)

d  = "You don't need to register. You're accepted. ..."
dv = model.encode(d)

v1.dot(dv)   # 0.32 — related

q2 = "How to install Docker on Windows?"
v2 = model.encode(q2)
v2.dot(dv)   # 0.01 — unrelated
```

**Why dot product = cosine similarity here:** `all-MiniLM-L6-v2` outputs
normalized vectors, so for unit vectors `dot product == cosine similarity`.
In practice scores rarely go below 0 — the model maps text into a region
where most vectors share positive components; there's no natural "opposite
meaning" vector.

---

## 3. Embedding Our Dataset

Reuses `ingest.py` from Module 1:
```bash
wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/ingest.py
```
```python
from ingest import load_faq_data
documents = load_faq_data()
```

Concatenate question + answer per document so a query can match either:
```python
texts = [doc["question"] + " " + doc["answer"] for doc in documents]
```

Embed in batches (full-dataset encoding is slow with no GPU, so chunk it):
```python
from tqdm.auto import tqdm

batch_size = 50
vectors = []
for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    vectors.extend(model.encode(batch))
```

Stack into a matrix — rows = documents, columns = dimensions:
```python
import numpy as np
X = np.array(vectors)   # shape: (1208, 384)
```

---

## 4. Vector Search (with numpy)

Score every document against the query in one matrix-vector multiply:
```python
query = "Can I still join the course after the start date?"
v_query = model.encode(query)
scores = X.dot(v_query)   # cosine similarity per document
```
(Equivalent to, but far faster than, a Python loop: `[v_query.dot(X[i]) for i in range(len(X))]` — numpy runs optimized C code.)

**Best match:**
```python
idx = np.argmax(scores)
documents[idx]
```

**Top-5 matches** (negate scores to turn a min-sort into a max-sort):
```python
top5 = np.argsort(-scores)[:5]
for idx in top5:
    print(scores[idx], documents[idx])
```

We return 5, not 1, because the right answer may be split across documents,
or the best match might rank second. 5 is a starting guess — a later
evaluation module helps tune it (3 vs. 10, etc.).

This brute-force numpy approach is fine for small datasets but doesn't
scale or support filtering — hence the move to a dedicated library next.

---

## 5. Vector Search with minsearch

[minsearch](https://github.com/alexeygrigorev/minsearch) (used for text
search in Module 1) also has a `VectorSearch` class, sharing the same API
(`fit`, `search`, `filter_dict`).

**Index:**
```python
from minsearch import VectorSearch

vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(X, documents)
```

**Search:**
```python
query_vector = model.encode("I just discovered the course. Can I still join it?")
results = vindex.search(query_vector, num_results=5)
```

**Filter by course** (so a user only sees answers from their own course):
```python
results = vindex.search(
    query_vector,
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)
```

---

## 6. RAG with Vector Search

Module 1's RAG pipeline has three swappable steps:
```python
def rag(question):
    search_results = search(question)
    user_prompt = build_prompt(question, search_results)
    return llm(user_prompt)
```
Only `search` changes — `build_prompt` and `llm` stay the same, because RAG
logic lives in a reusable [`RAGBase`](../01-agentic-rag/code/rag_helper.py) class.

```bash
wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/rag_helper.py
wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/ingest.py
```

Since vector search needs a *vector* query (not raw text), subclass
`RAGBase` and override `search` to embed the query first:
```python
class RAGVector(RAGBase):
    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}
        return self.index.search(
            query_vector, num_results=num_results, filter_dict=filter_dict
        )
```

```python
vector_assistant = RAGVector(embedder=model, index=vindex, llm_client=openai_client)
vector_assistant.rag("the program has already begun, can I still sign up?")
```

Same trick (override one method) works for swapping LLM providers later.

---

## 7. Vector Search with sqlitesearch

minsearch's vector search has three real limits:
- Re-embeds the whole dataset on every startup
- Keeps everything in memory
- Brute-force search (compares query against *every* document)

These don't matter at ~1,000 documents, but do beyond ~10,000. That's where
**ANN (Approximate Nearest Neighbor)** search comes in — narrow to a likely
region first, then score only within it (a small accuracy trade for a big
speed gain), vs. **exact NN** which always scores everything.

```text
NN (exact):    compare query against ALL documents -> top 5
ANN (approx):  narrow down to a region -> compare within region -> top 5
```

**sqlitesearch** is minsearch's persistent sibling — stores vectors in
SQLite on disk, supports ANN, lets one process write and another read.
```bash
uv add sqlitesearch
```

**ANN modes:**
| Mode | Scale | Method |
|---|---|---|
| `lsh` (default) | up to 100K vectors | random hyperplane projections |
| `ivf` | 10K–500K vectors | K-means clustering |
| `hnsw` | 10K–1M+ vectors | proximity graph (highest recall) |

All modes do two-phase search: approximate candidates → exact cosine rerank.

**Index & search:**
```python
from sqlitesearch import VectorSearchIndex

vs_index = VectorSearchIndex(keyword_fields=["course"], mode="ivf", db_path="faq_vectors2.db")
vs_index.fit(vectors, documents)

query_vector = model.encode("I just discovered the course. Can I still join it?")
results = vs_index.search(query_vector, num_results=5)
results = vs_index.search(query_vector, filter_dict={"course": "llm-zoomcamp"}, num_results=5)

vs_index.close()
```

**Reopen later without re-embedding** (one process ingests, another queries —
this matters here because embedding ~1200 docs takes ~a minute, and you don't
want users waiting on that at app startup):
```python
model = SentenceTransformer("all-MiniLM-L6-v2")
vs_index = VectorSearchIndex(keyword_fields=["course"], mode="ivf", db_path="faq_vectors2.db")
# no .fit() needed — index already on disk
results = vs_index.search(model.encode("How do I run Kafka?"), num_results=5)
```

Plug into RAG the same way as minsearch — reuse the `RAGVector` class from
lesson 6, just pass `vs_index` instead of `vindex`.

**minsearch vs. sqlitesearch:**

| | minsearch `VectorSearch` | sqlitesearch `VectorSearchIndex` |
|---|---|---|
| Storage | In-memory (numpy) | Persistent SQLite `.db` |
| Search | Exact cosine | ANN (LSH/IVF/HNSW) + exact rerank |
| Startup | Must re-embed every time | Can reopen an existing index |
| Best for | Experiments, notebooks | Pet projects, persistence |

sqlitesearch's niche: it needs only SQLite + numpy, so it runs anywhere a
free SQLite DB is available and a dedicated vector DB would cost extra. For
most production work, reach for something else — covered next.

---

## 8. Vector Search with PGVector

Many real databases support vector search (Elasticsearch, Qdrant, Chroma).
This lesson uses **Postgres** via the [pgvector](https://github.com/pgvector/pgvector)
extension — common in production stacks already, plus you get transactions
and concurrent access for free.

**Run Postgres + pgvector in Docker:**
```bash
docker run -it \
    --name pgvector \
    -e POSTGRES_USER=user \
    -e POSTGRES_PASSWORD=pswd \
    -e POSTGRES_DB=faq \
    -v pgvector_data:/var/lib/postgresql/data \
    -p 5432:5432 \
    pgvector/pgvector:pg17
```

**Python driver** (psycopg v3 — supports `conn.execute()` directly, unlike psycopg2):
```bash
uv add psycopg[binary]      # zsh: uv add 'psycopg[binary]'
```

**Connect & enable the extension:**
```python
import psycopg

conn = psycopg.connect("postgresql://user:pswd@localhost:5432/faq")
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
```

**Create table** with a `vector(384)` column matching the embedding model's dimensionality:
```python
conn.execute("DROP TABLE IF EXISTS documents")
conn.execute("""
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        course TEXT,
        section TEXT,
        question TEXT,
        answer TEXT,
        embedding vector(384)
    )
""")
```

**Insert** (vectors go in as text, cast with `::vector`):
```python
def vec_to_str(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"

for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):
    conn.execute(
        """
        INSERT INTO documents (course, section, question, answer, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        """,
        (doc["course"], doc["section"], doc["question"], doc["answer"], vec_to_str(vec))
    )
conn.commit()
```

**Search with cosine distance** (`<=>` operator; ascending distance = most similar first):
```python
results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_str, query_str)
).fetchall()
```

**Filter by course** — just plain SQL `WHERE`:
```python
results = conn.execute(
    """
    SELECT course, question, answer, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents WHERE course = %s
    ORDER BY embedding <=> %s::vector LIMIT 5
    """,
    (query_str, "llm-zoomcamp", query_str)
).fetchall()
```

**Speed up at scale** with an HNSW index (same algorithm dedicated vector DBs use):
```python
conn.execute("CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)")
```

**RAG integration** — same override pattern, `index=None` since `RAGBase` expects an index object:
```python
class RAGPgVector(RAGBase):
    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_str = vec_to_str(query_vector)
        rows = self.conn.execute(
            """
            SELECT course, section, question, answer FROM documents
            WHERE course = %s ORDER BY embedding <=> %s::vector LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()
        return [{"course": r[0], "section": r[1], "question": r[2], "answer": r[3]} for r in rows]
```

**Tool comparison:**

| | minsearch | sqlitesearch | PGVector |
|---|---|---|---|
| Setup | None | None | Docker + Postgres |
| Storage | In-memory | SQLite file | Postgres DB |
| Scale | Small | Pet-project scale | Millions of records |
| Best for | Notebooks, experiments | Pet projects | Production systems |

Reach for PGVector when you need concurrent reads/writes, transactions, or
integration with an existing Postgres-based app.

---

## 9. ONNX Embedder (optional)

Production wants minimal dependencies. `sentence-transformers` pulls in
PyTorch + CUDA libraries; **ONNX Runtime** serves the same model without
that weight.

Measured virtual-env sizes for the same task:

| Library | Size | Packages |
|---|---|---|
| sentence-transformers | 4.8 GB | 58 |
| ONNX Runtime | 147 MB | 27 |

That's **~33x smaller** for identical embeddings/results. Use
sentence-transformers for dev/experiments; switch to ONNX for production.

**Setup:**
```bash
mkdir llm-zoomcamp-onnx && cd llm-zoomcamp-onnx
uv init --no-workspace
uv add onnxruntime tokenizers numpy tqdm minsearch
uv add --dev huggingface-hub jupyter
uv run python -m ipykernel install --user --name llm-zoomcamp-onnx --display-name "llm-zoomcamp-onnx"
```

**Download the ONNX model** (one-time, via [embed/download.py](../embed/download.py)):
```bash
uv run python download.py
```
Produces `models/Xenova/all-MiniLM-L6-v2/{tokenizer.json, model.onnx}`. Add `models/` to `.gitignore`.

**The `Embedder` class** ([embed/embedder.py](../embed/embedder.py)) gives the same `encode` interface as before. Under the hood: tokenize → run ONNX graph on CPU → mean-pool token embeddings (weighted by attention mask) → L2-normalize.

```python
from embedder import Embedder

embed = Embedder()
v1 = embed.encode("Can I still join the course after the start date?")
# same dot-product comparisons, same results, no PyTorch
```

Batch embedding uses `embed.encode_batch(batch)` in place of `model.encode(batch)` — everything else (FAQ loading, batching, numpy search) is identical to earlier lessons.

**Other ONNX models available** (swap the name in `download.py` + the path in `Embedder()`):
- `Xenova/all-MiniLM-L6-v2` (384d) — best small general-purpose
- `Xenova/all-MiniLM-L12-v2` (384d) — better quality, slower
- `Xenova/paraphrase-MiniLM-L6-v2` (384d) — paraphrase detection
- `Xenova/paraphrase-multilingual-MiniLM-L12-v2` (384d) — multilingual
- `Xenova/multilingual-e5-small` (384d) / `multilingual-e5-base` (768d) — multilingual retrieval
- `Xenova/bge-small-en-v1.5` (384d) / `bge-base-en-v1.5` (768d) — strong retrieval
- `Xenova/gte-small` (384d) / `gte-base` (768d) — lightweight modern models

Since the runtime only needs `onnxruntime`, `tokenizers`, and `numpy`, this
is deployable to small Docker images, serverless functions, and edge devices.

---

## 10. Next Steps

**What this module covered:**
- What embeddings are and how they turn text into vectors
- Generating embeddings for the FAQ dataset with sentence-transformers
- Vector search via numpy, minsearch, sqlitesearch, and PGVector
- Wiring vector search into RAG via the `RAGVector` class

**When to actually use vector search** — it adds real overhead (embedding
model, computing/storing embeddings, encoding every query), so don't take it
on without a reason. Recommended progression:

1. **v1:** Start with text search — get RAG working end-to-end (often handles most questions fine on its own).
2. **v2:** Add vector search once you can *measure* that text search misses relevant results (typically when users phrase questions differently than your docs).
3. **v3:** Combine both as **hybrid search** (text + vector), which typically outperforms either alone.

Move between versions based on evaluation results, not assumption — a later
module covers measuring search quality so you can tell a marginal gain from
a real one.

**Hybrid search & reranking:** once both text and vector search exist, merge
results (e.g. Reciprocal Rank Fusion) or rerank candidates with a separate
relevance model. Covered in the *Hybrid Search* lesson, Best Practices module.

**Try next:**
- Compare text vs. vector search on your own data
- Experiment with different embedding models
- Try PGVector with a larger dataset
- Try another vector DB — Elasticsearch, Qdrant, Weaviate, Chroma (same core concepts: embed, store, search by similarity)
- Evaluate your search results (covered in a later module)
