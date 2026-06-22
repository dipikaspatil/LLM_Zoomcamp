import json
from openai import OpenAI
import minsearch
from rag_helper import RAGBase
from gitsource import chunk_documents

# Load documents
with open("data/documents.json") as f:
    documents = json.load(f)

chunks = chunk_documents(documents, size=2000, step=1000)

# Index the chunks
index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)
index.fit(chunks)

# Run RAG with the chunk index
client = OpenAI()
rag = RAGBase(index=index, llm_client=client)

query = "How does the agentic loop keep calling the model until it stops?"
answer, usage = rag.rag(query)

print("Answer:", answer)
print("Input tokens (chunked):", usage.input_tokens)
print("Input tokens (Q3 full docs):", 7110)
print("Ratio:", round(7110 / usage.input_tokens, 1), "× fewer")