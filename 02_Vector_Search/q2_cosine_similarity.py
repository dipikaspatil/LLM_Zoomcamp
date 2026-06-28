from embedder import Embedder
from gitsource import GithubRepositoryDataReader

embed = Embedder()
query = "How does approximate nearest neighbor search work?"
v = embed.encode(query)


# Get raw text
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "02-vector-search/lessons/07-sqlitesearch-vector.md" in path,
)

document = [file.parse() for file in reader.read()]

#print(document[0].get("content"))

# Embed document/text
d = embed.encode(document[0].get("content"))

# Get dot product/cosine similarity
similarity = v.dot(d)
print(similarity)
