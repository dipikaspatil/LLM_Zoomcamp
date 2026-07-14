from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from gitsource import GithubRepositoryDataReader
from evaluation_utils import llm_structured

load_dotenv()          # reads OPENAI_API_KEY (and any other secrets) from a .env file
client = OpenAI()


# Structured output schema — forces the LLM to return JSON matching this shape
# instead of free-form text we'd have to parse ourselves.
class Questions(BaseModel):
    questions: list[str]


# "developer" role message — persistent instructions the model follows on every call.
# Kept separate from the per-document content so the same instructions can be reused
# across all 3 documents without repeating them in every prompt.
INSTRUCTIONS = """
You are a student taking this course. Based on the lesson content provided,
formulate 5 questions this student might ask.

The questions should be complete and not too short.
Use as few exact words from the lesson content as possible — write them
in your own words, the way a real student would phrase them.
""".strip()

# Template for the "user" role message — filled in per document with the actual
# filename and lesson content before being sent to the model.
USER_PROMPT_TEMPLATE = """
FILENAME: {filename}

LESSON CONTENT:
{content}
""".strip()


# Downloads the repo at a pinned commit (so results are reproducible even if the
# course repo changes later) and keeps only markdown files under any "lessons/" folder.
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]  # each doc now has filename + content

input_tokens = []

# Q1 only asks for the first 3 lesson pages — not the full 72-page / 295-chunk
# dataset used later in Q2-Q6, and note this loops over whole documents,
# not chunks (chunking isn't needed just to generate questions from a page).
for doc in documents[:3]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        filename=doc["filename"],
        content=doc["content"],
    )

    # llm_structured() (from evaluation_utils.py) sends instructions + user_prompt,
    # parses the response into a Questions object, and also returns token usage
    # so we can inspect exactly what each call cost.
    parsed, usage = llm_structured(client, INSTRUCTIONS, user_prompt, Questions)

    print(doc["filename"], "->", usage.input_tokens, "input tokens")
    input_tokens.append(usage.input_tokens)

# The actual Q1 answer: average prompt/input token count across the 3 calls.
average_input_tokens = sum(input_tokens) / len(input_tokens)
print("\nAverage input tokens:", average_input_tokens)
