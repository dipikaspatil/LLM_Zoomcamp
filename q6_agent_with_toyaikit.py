import json
from openai import OpenAI
import minsearch
from gitsource import chunk_documents

from toyaikit.tools import Tools
from toyaikit.llm import OpenAIClient
from toyaikit.chat.runners import OpenAIResponsesRunner

# Load chunks and build index
with open("data/documents.json") as f:
    documents = json.load(f)

chunks = chunk_documents(documents, size=2000, step=1000)

index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)
index.fit(chunks)

# Search function with counter
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

# Register tool
tools = Tools()
tools.add_tool(search_course)

# Build runner
INSTRUCTIONS = "You're a course teaching assistant. Answer the student's question using the search tool. Make multiple searches with different keywords before answering."

llm_client = OpenAIClient(model="gpt-5.4-mini", client=OpenAI())

runner = OpenAIResponsesRunner(
    tools=tools,
    developer_prompt=INSTRUCTIONS,
    llm_client=llm_client,
)

# Run the agent
question = "How does the agentic loop work, and how is it different from plain RAG?"
result = runner.loop(question)

print(f"\nSearch tool called {search_count} times")
