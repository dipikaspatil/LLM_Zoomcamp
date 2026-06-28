from gitsource import GithubRepositoryDataReader
from gitsource import chunk_documents
from embedder import Embedder
import numpy as np
from minsearch import VectorSearch
from minsearch import Index

# Get documents
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

# Chunk documents
chunks = chunk_documents(documents, size=2000, step=1000)

# Embed every chunk in one batch call
texts = [c["content"] for c in chunks]

embed = Embedder()

vectors = embed.encode_batch(texts)

# Below line converts the embeddings from a plain Python list into a 2D matrix that numpy can do fast math on.
# Before this line: vectors is a list of 295 separate items (one per chunk), where each item is itself an array of 384 numbers. Think of it as 295 separate sticky notes, each with 384 numbers written on it.
# np.array(vectors) stacks all 295 of those sticky notes into one single table (a matrix):
X = np.array(vectors)   # shape: (295, 384)

'''
X = 
[ [ -0.02, 0.11, ... 384 numbers ... ],   <- chunk 0's vector
  [  0.05, -0.03, ... 384 numbers ... ],  <- chunk 1's vector
  ...
  [  ... ] ]                               <- chunk 294's vector
shape: (295, 384) describes the table's dimensions:

295 rows → one row per chunk (you have 295 chunks total)
384 columns → one column per embedding dimension (fixed by the model, all-MiniLM-L6-v2)

Why bother converting it
Once it's a proper numpy matrix, you can compare your query vector v against all 295 chunks in a single operation:


scores = X.dot(v)

This does 295 dot products at once (one per row) using optimized C code under the hood — much faster than writing a Python for loop like [v.dot(vec) for vec in vectors]. 
The result, scores, is a flat array of 295 numbers — one similarity score per chunk — which is exactly what scores.argmax() and np.argsort(-scores) then operate on to find your top matches.

'''
# Vector search
# Embed the new query
query = "How do I give the model access to tools?"
qv = embed.encode(query)

vindex = VectorSearch()
vindex.fit(X, chunks)

# Search
vector_results = vindex.search(qv, num_results=5)
# print filename
print("Top 5 results of vector search")
for i in range(len(vector_results)):
    print(f"rank {i}: {vector_results[i]["filename"]}")
print("\n")

# Text search
tindex = Index(text_fields=["content"])
tindex.fit(chunks)
text_results = tindex.search(query, num_results=5)   # text search takes the raw string directly
# print filename
print("Top 5 results of text search")
for i in range(len(text_results)):
    print(f"rank {i}: {text_results[i]["filename"]}")
print("\n")

def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])   # identifies one specific chunk
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

results = rrf([vector_results, text_results])

print("Top 5 results of RRF fused")
for i in range(len(results)):
    print(f"{i}: {results[i]["filename"]}")