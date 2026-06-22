import json
from openai import OpenAI
import minsearch
from gitsource import chunk_documents

# Load chunks and build index (same as Q5)
with open("data/documents.json") as f:
    documents = json.load(f)

chunks = chunk_documents(documents, size=2000, step=1000)

index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)
index.fit(chunks)

# Search function with call counter
search_count = 0

def search_course(query: str) -> str:
    """Search the course content for relevant information about LLM and RAG topics."""
    global search_count
    search_count += 1
    print(f"  [Search #{search_count}] Query: {query}")

    results = index.search(query, num_results=5)
    lines = []
    for doc in results:
        lines.append(doc['filename'])
        lines.append(doc['content'])
        lines.append('')
    return '\n'.join(lines).strip()

# Tool schema for OpenAI
tools = [{
    "type": "function",
    "name": "search_course",
    "description": "Search the course content for relevant information about LLM and RAG topics.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
}]

INSTRUCTIONS = "You're a course teaching assistant. Answer the student's question using the search tool. Make multiple searches with different keywords before answering."

client = OpenAI()
messages = [
    {"role": "developer", "content": INSTRUCTIONS},
    {"role": "user", "content": "How does the agentic loop work, and how is it different from plain RAG?"}
]

# Agentic loop
while True:
    response = client.responses.create(
        model="gpt-5.4-mini",
        input=messages,
        tools=tools,
    )

    function_calls = [item for item in response.output if item.type == "function_call"]

    if not function_calls:
        break

    # Append assistant output to message history
    messages += response.output

    # Execute each tool call and append results
    for fc in function_calls:
        args = json.loads(fc.arguments)
        result = search_course(args["query"])
        messages.append({
            "type": "function_call_output",
            "call_id": fc.call_id,
            "output": result
        })

print("\nAnswer:", response.output_text)
print(f"\nSearch tool called {search_count} times")
