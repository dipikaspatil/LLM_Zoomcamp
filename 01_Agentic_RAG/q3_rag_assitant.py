import json
from openai import OpenAI
import minsearch
from rag_helper import RAGBase

# Load documents
with open("data/documents.json") as f:
    documents = json.load(f)

# Build index
index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)
index.fit(documents)

# Run RAG
client = OpenAI()
rag = RAGBase(index=index, llm_client=client)

query = "How does the agentic loop keep calling the model until it stops?"
answer, usage = rag.rag(query)

print("Answer:", answer)
print("Input tokens:", usage.input_tokens)