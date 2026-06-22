import minsearch, json

# Build the index
index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)

# Get all documents
with open("data/documents.json") as f:
    documents = json.load(f)

# Index all documents
index.fit(documents)

# Search
query = "How does the agentic loop keep calling the model until it stops?"
results = index.search(query)

print(f"First result filename: {results[0]['filename']}")
