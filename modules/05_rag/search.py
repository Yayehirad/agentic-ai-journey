"""Module 05: find the chunks in ChromaDB that best match a question."""
import sys
from pathlib import Path

import chromadb
from ollama import embed

# Must match ingest.py, so we open the same database and collection
DB_DIR = Path(__file__).parent / "db"
COLLECTION = "notes"
EMBED_MODEL = "nomic-embed-text"

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection(
    COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None)


def search(question: str, k: int = 4) -> list[dict]:
    """Return the k closest chunks as dicts with file, page, text, distance."""
    if collection.count() == 0:
        return []    # nothing ingested yet

    # Questions get "search_query: " (stored text got "search_document: ").
    # Both use the SAME model, so their numbers can be compared.
    vector = embed(model=EMBED_MODEL,
                   input=f"search_query: {question}").embeddings[0]
    result = collection.query(query_embeddings=[vector], n_results=k)

    # Chroma returns one list per question; we asked one question, so [0]
    return [
        {"file": meta["file"], "page": meta["page"],
         "text": text, "distance": dist}
        for text, meta, dist in zip(result["documents"][0],
                                    result["metadatas"][0],
                                    result["distances"][0])
    ]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    if not question:
        print('Usage: python search.py "your question"')
        sys.exit(1)

    hits = search(question)
    if not hits:
        print("No chunks stored yet. Run ingest.py first.")
    # Distance: 0 = same meaning, bigger = less related
    for i, hit in enumerate(hits, start=1):
        print(f"\n#{i}  {hit['file']}, page {hit['page']}  "
              f"(distance {hit['distance']:.3f})")
        print(hit["text"])
