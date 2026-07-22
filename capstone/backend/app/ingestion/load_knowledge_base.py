"""
One-off script: reads knowledge base source files, splits them into chunks,
embeds each chunk, and uploads everything into Qdrant.

Expected folder layout (create these yourself under data/knowledge_base/):
    data/knowledge_base/tactical_concepts/*.md
    data/knowledge_base/world_cup_history/*.md

The folder name becomes the "section" tag on every chunk, which is how
search_rag() later filters results to just the section the user picked in the UI.

Run manually whenever you add or change knowledge base content:
    python -m app.ingestion.load_knowledge_base
"""
import pathlib
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import settings
from app.tools.vector_store import COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_DIM

# Resolve data/knowledge_base relative to this file, so the script works
# no matter what directory you run it from
KNOWLEDGE_BASE_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "knowledge_base"


def load_documents() -> list[dict]:
    """Walk data/knowledge_base/<section>/*.md and return each file's raw text + its section tag."""
    documents = []

    for section_dir in KNOWLEDGE_BASE_DIR.iterdir():
        if not section_dir.is_dir():
            continue  # skip stray files directly in knowledge_base/, only folders count as sections

        section = section_dir.name  # e.g. "tactical_concepts"

        for file_path in section_dir.glob("*.md"):
            documents.append({
                "text": file_path.read_text(encoding="utf-8"),
                "section": section,
                "source": file_path.name,  # kept so answers can cite which file a fact came from
            })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split each document into smaller overlapping pieces.

    Why chunk at all: embedding a whole file and comparing it to a short
    question gives a fuzzy, diluted match. Chunking lets retrieval return the
    specific paragraph that actually answers the question.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,     # characters per chunk — small enough to stay focused, big enough for context
        chunk_overlap=100,  # slight overlap so a sentence split across two chunks isn't lost entirely
    )

    chunks = []
    for doc in documents:
        for piece in splitter.split_text(doc["text"]):
            chunks.append({
                "text": piece,
                "section": doc["section"],
                "source": doc["source"],
            })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Call OpenAI once per batch to turn every chunk's text into a vector, then attach it to the chunk."""
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)

    texts = [chunk["text"] for chunk in chunks]
    vectors = embedder.embed_documents(texts)  # batches internally — much cheaper than one call per chunk

    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector

    return chunks


def upload_to_qdrant(chunks: list[dict]) -> None:
    """(Re)create the Qdrant collection and upload every chunk as a point."""
    client = QdrantClient(url=settings.QDRANT_URL)

    # recreate_collection wipes any existing data first — fine here since we always
    # rebuild the whole knowledge base from source files, never patch it incrementally
    # recreate_collection is deprecated — explicitly check-then-recreate instead
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),  # Qdrant requires a unique id per point; content itself isn't the id
            vector=chunk["vector"],
            payload={  # payload = the metadata returned alongside a search hit
                "text": chunk["text"],
                "section": chunk["section"],
                "source": chunk["source"],
            },
        )
        for chunk in chunks
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Uploaded {len(points)} chunks to Qdrant collection '{COLLECTION_NAME}'")


def main():
    documents = load_documents()
    print(f"Loaded {len(documents)} source documents")

    chunks = chunk_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print("Generated embeddings for all chunks")

    upload_to_qdrant(chunks)


if __name__ == "__main__":
    main()
