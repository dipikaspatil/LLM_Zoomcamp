"""
Shared settings for the Qdrant knowledge base collection.

Both the ingestion script (load_knowledge_base.py) and the runtime search
tool (rag_search.py) import from here, so they always agree on the collection
name and embedding model — if this were duplicated in both files, changing
the embedding model in one place and forgetting the other would silently
break search (vectors of different sizes/models can't be compared).
"""

# Name of the Qdrant collection where all knowledge base chunks live
COLLECTION_NAME = "soccermind_knowledge"

# Must match the OpenAI embedding model used everywhere in the project
EMBEDDING_MODEL = "text-embedding-3-small"

# text-embedding-3-small always outputs vectors of this length —
# Qdrant needs to know this upfront to allocate the collection correctly
EMBEDDING_DIM = 1536
