from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch
from embedder import Embedder
import pandas as pd

# Path to the local ONNX model downloaded via download.py — same model used in Q2/Q3,
# so chunk vectors and query vectors stay in the same embedding space.
EMBED_MODEL_PATH = "models/Xenova/all-MiniLM-L6-v2"

# Downloads the repo at a pinned commit and keeps only markdown files under any
# "lessons/" folder — same 72-page dataset used across Q2-Q6.
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()] # each doc now has filename + content

chunks = chunk_documents(documents, size=2000, step=1000)

# --- vector search, using the same ONNX embedder as Module 02 ---
embedder = Embedder(path=EMBED_MODEL_PATH)

print(f"Embedding {len(chunks)} chunks...")
chunk_texts = [chunk["content"] for chunk in chunks]
chunk_vectors = embedder.encode_batch(chunk_texts)   # one batched call instead of 295 API calls

vector_index = VectorSearch(keyword_fields=["filename"])
vector_index.fit(chunk_vectors, chunks)


def vector_search(query, num_results=5):
    # Query must be embedded with the *same* model as the chunks — otherwise
    # cosine similarity is comparing vectors from two unrelated spaces.
    query_vector = embedder.encode(query)
    return vector_index.search(query_vector, num_results=num_results)


# Turns one query's search results into a list of 1s/0s — 1 where a result's
# filename matches the ground-truth filename, 0 otherwise. Position matters here:
# index 0 is rank 1, which mrr() below uses directly.
def compute_relevance(results, expected_filename):
    return [1 if r["filename"] == expected_filename else 0 for r in results]


# Runs every ground-truth question through search_function and collects one
# relevance list per question — same harness reused from Q4, just pointed at
# vector_search instead of text_search.
def evaluate(ground_truth, search_function):
    relevance_total = []
    for _, row in ground_truth.iterrows():
        results = search_function(row["question"])
        relevance_total.append(compute_relevance(results, row["filename"]))
    return relevance_total


# Mean Reciprocal Rank — unlike Hit Rate, this rewards ranking the correct
# document *higher*: score = 1/rank for the first match found, 0 if no match at all.
def mrr(relevance_total):
    total = 0.0
    for line in relevance_total:
        for rank, is_relevant in enumerate(line, start=1):
            if is_relevant:
                total += 1 / rank
                break
    return total / len(relevance_total)


# Q5
# Must be the official downloaded file — same ground truth used in Q4, so Hit Rate
# (text) and MRR (vector) results are comparable across the same 360 questions.
ground_truth = pd.read_csv("ground-truth.csv")
relevance = evaluate(ground_truth, vector_search)
print(f"MRR for vector_search across all 360 questions: {mrr(relevance)}")
