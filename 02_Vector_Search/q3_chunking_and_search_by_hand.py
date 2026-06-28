from gitsource import GithubRepositoryDataReader
from gitsource import chunk_documents
from embedder import Embedder
import numpy as np

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


# Score every chunk against the Q1 query vector
query = "How does approximate nearest neighbor search work?"
v = embed.encode(query)

scores = X.dot(v)
# idx = scores.argmax()
# print(chunks[idx]["filename"])

# Get top 5 chunk and score
# np.argsort(-scores) sorts indices by ascending negated score, 
# which is the same as sorting by descending original score (the "negate to flip min-sort into max-sort" trick from the earlier numpy lesson).
# [:5] takes the first 5 of that sorted order — your top 5 matches.
top5 = np.argsort(-scores)[:5]

# The loop just prints each one's score (rounded for readability) alongside which file it came from.
for i in top5:
    print(round(scores[i], 4), chunks[i]["filename"])
