"""Chroma setup, seeding, and retrieval helpers for the Flask RAG API lesson."""

from typing import Any, Dict, List

import chromadb

from ai_client import get_embedding
from documents import FACILITY_DOCUMENTS


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "facility_documents"
TOP_K = 3


# def get_collection(reset: bool = False):
#     """Return the lesson Chroma collection."""
#     # TODO: Step 5 - create a PersistentClient and get or reset the collection.
#     raise NotImplementedError("Complete get_collection() in Step 5.")

def get_collection(reset: bool = False):
    """Return the lesson Chroma collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def seed_collection() -> int:
    """Store the facility documents, embeddings, and metadata in Chroma."""
    collection = get_collection(reset=True)
    ids = [document["id"] for document in FACILITY_DOCUMENTS]
    documents = [
        f"{document['title']}. {document['text']}"
        for document in FACILITY_DOCUMENTS
    ]
    metadatas = [
        {
            "title": document["title"],
            "category": document["category"],
        }
        for document in FACILITY_DOCUMENTS
    ]
    embeddings = [get_embedding(document_text) for document_text in documents]
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return collection.count()



def query_collection(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Retrieve the most relevant Chroma results for a user question."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = get_embedding(question)
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    ids = raw_results.get("ids") or [[]]

    documents = raw_results.get("documents") or [[]]

    metadatas = raw_results.get("metadatas") or [[]]

    distances = raw_results.get("distances") or [[]]

    results: List[Dict[str, Any]] = []

    for index, document_id in enumerate(ids[0]):

        metadata = metadatas[0][index] or {}

        distance = float(distances[0][index])

        score = 1 - distance

        results.append(

            {
                "id": document_id,
                "title": metadata.get("title", "Untitled source"),
                "category": metadata.get("category", "uncategorized"),
                "text": documents[0][index],
                "score": score,
            }
        )

    return results
