from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
import pandas as pd

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
# Same index construction as Q2/Q3, rebuilt here so this script can run standalone.
text_index = Index(text_fields=["content"], keyword_fields=["filename"])
text_index.fit(chunks)


def text_search(query, num_results=5):
    return text_index.search(query, num_results=num_results)


# Turns one query's search results into a list of 1s/0s — 1 where a result's
# filename matches the ground-truth filename, 0 otherwise. Position in the list
# matters: index 0 is rank 1, used later by MRR (not needed for Hit Rate itself).
def compute_relevance(results, expected_filename):
    return [1 if r["filename"] == expected_filename else 0 for r in results]


# Runs every ground-truth question through search_function and collects one
# relevance list per question — the raw material both hit_rate() and mrr() consume.
def evaluate(ground_truth, search_function):
    relevance_total = []
    for _, row in ground_truth.iterrows():
        results = search_function(row["question"])
        relevance_total.append(compute_relevance(results, row["filename"]))
    return relevance_total


# Fraction of queries where the correct document appears *anywhere* in the top-k —
# ignores rank position entirely (unlike MRR in Q5).
def hit_rate(relevance_total):
    return sum(1 for line in relevance_total if 1 in line) / len(relevance_total)


# Q4
# Must be the official downloaded file, not self-generated — the question wording
# is what the MCQ answer options are calibrated against.
ground_truth = pd.read_csv("ground-truth.csv")

relevance = evaluate(ground_truth, text_search)
print(f"Hit Rate for text_search across all 360 questions: {hit_rate(relevance)}")
