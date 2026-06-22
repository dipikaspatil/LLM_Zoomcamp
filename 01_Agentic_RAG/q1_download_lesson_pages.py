from gitsource import GithubRepositoryDataReader
import os, json

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []
for file in files:
    doc = file.parse()
    documents.append(doc)

os.makedirs("data", exist_ok=True)
with open("data/documents.json", "w") as f:
    json.dump(documents, f)

print(f"Number of lesson pages: {len(documents)}")