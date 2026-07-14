from gitsource import GithubRepositoryDataReader, chunk_documents
import pandas as pd
from minsearch import Index, VectorSearch
from embedder import Embedder

# Path to the local ONNX model downloaded via download.py (same model Module 02 used) —
# reusing it here keeps chunk vectors and query vectors in the same embedding space.
EMBED_MODEL_PATH = "models/Xenova/all-MiniLM-L6-v2"

# Downloads the repo at a pinned commit and keeps only markdown files under any
# "lessons/" folder — same 72-page dataset used for the rest of the homework (Q2-Q6).
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()] # each doc now has filename + content

# Unlike Q1 (whole pages), search runs over overlapping 2000-char chunks —
# a full page mixes too many topics into one match target for search to be precise.
chunks = chunk_documents(documents, size=2000, step=1000)

# --- text search (TF-IDF / keyword-style, over chunk content) ---
# text_fields = what gets matched against the query; keyword_fields = carried through
# on each result so we can read back which file a chunk came from.
text_index = Index(text_fields=["content"], keyword_fields=["filename"])
text_index.fit(chunks)

def text_search(query, num_results=5):
    return text_index.search(query, num_results=num_results)

# --- vector search, now using the same ONNX embedder as Module 02 ---
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


# --- Q2 / Q3 ---
# Row 0 of the official ground truth — same question run through both search
# methods, to compare keyword-based vs. embedding-based retrieval directly.
ground_truth = pd.read_csv("ground-truth.csv")
first_question = ground_truth.iloc[0]["question"]
print("\nQuestion:", first_question)

text_results = text_search(first_question, num_results=5)
print("Q2 - text_search top result:", text_results[0]["filename"])

vector_results = vector_search(first_question, num_results=5)
print("Q3 - vector_search top result:", vector_results[0]["filename"])
