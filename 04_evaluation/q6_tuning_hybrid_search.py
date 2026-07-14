from gitsource import GithubRepositoryDataReader, chunk_documents
import pandas as pd
from minsearch import Index, VectorSearch
from embedder import Embedder

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

# --- text search (TF-IDF / keyword-style, over chunk content) ---
text_index = Index(text_fields=["content"], keyword_fields=["filename"])
text_index.fit(chunks)


def text_search(query, num_results=5):
    return text_index.search(query, num_results=num_results)


# --- vector search, using the same ONNX embedder as Module 02 ---
embedder = Embedder(path=EMBED_MODEL_PATH)

print(f"Embedding {len(chunks)} chunks...")
chunk_texts = [chunk["content"] for chunk in chunks]
chunk_vectors = embedder.encode_batch(chunk_texts)   # one batched call instead of 295 API calls

vector_index = VectorSearch(keyword_fields=["filename"])
vector_index.fit(chunk_vectors, chunks)


def vector_search(query, num_results=5):
    query_vector = embedder.encode(query)
    return vector_index.search(query_vector, num_results=num_results)


# Reciprocal Rank Fusion — combines two ranked result lists into one, using only
# rank position (not raw scores, which aren't comparable between TF-IDF and cosine
# similarity). A doc appearing near the top of *both* lists accumulates a bigger
# combined score than one appearing in only one list.
#
# k is a damping constant: small k makes the fusion very sensitive to exact rank
# (1/(k+1) vs 1/(k+2) differ a lot); large k flattens that sensitivity out, since
# 1/(k+rank) barely changes as rank varies once k dominates the denominator.
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            scores[doc["filename"]] = scores.get(doc["filename"], 0) + 1 / (k + rank)

    ranked_filenames = sorted(scores, key=scores.get, reverse=True)
    return ranked_filenames[:num_results]


# Note: unlike Q4/Q5's compute_relevance, `results` here is a list of bare
# filenames (rrf() returns strings, not result dicts) — so this compares
# directly instead of indexing into r["filename"].
def compute_relevance(results, expected_filename):
    return [1 if r == expected_filename else 0 for r in results]


# Pulls the top 10 from each individual search (deeper than the final top-5) so
# RRF has more candidates to actually fuse, then fuses and truncates to top-5.
def hybrid_search(query, k=60, num_results=5):
    text_results = text_search(query, num_results=10)     # pull a bit deeper than final top-5
    vector_results = vector_search(query, num_results=10)
    return rrf([text_results, vector_results], k=k, num_results=num_results)


# Same evaluation harness as Q4/Q5, generic over whatever search_function is passed in.
def evaluate(ground_truth, search_function):
    relevance_total = []
    for _, row in ground_truth.iterrows():
        results = search_function(row["question"])
        relevance_total.append(compute_relevance(results, row["filename"]))
    return relevance_total


def mrr(relevance_total):
    total = 0.0
    for line in relevance_total:
        for rank, is_relevant in enumerate(line, start=1):
            if is_relevant:
                total += 1 / rank
                break
    return total / len(relevance_total)


# Q6
ground_truth = pd.read_csv("ground-truth.csv")

for k in [1, 50, 100, 200]:
    # k=k as a default argument captures the *current* loop value immediately.
    # Without it, all four closures would share one `k` variable and end up
    # reading whatever it was set to last (200) by the time they actually run.
    def search_fn(query, k=k):
        return hybrid_search(query, k=k)

    relevance = evaluate(ground_truth, search_fn)
    print(f"Hybrid search: tuning the RRF parameter k: {k} , mrr: {mrr(relevance)}")
