from gitsource import chunk_documents
import json

# Load documents
with open("data/documents.json") as f:
    documents = json.load(f)

chunks = chunk_documents(documents, size=2000, step=1000)

print(f"Number of chunks: {len(chunks)}" )