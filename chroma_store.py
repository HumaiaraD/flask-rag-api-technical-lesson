"""Chroma setup, seeding, and retrieval helpers for the Flask RAG API lesson."""

from typing import Any, Dict, List

import chromadb

from ai_client import get_embedding
from documents import FACILITY_DOCUMENTS


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "facility_documents"
TOP_K = 3


def get_collection(reset: bool = False):
    """Return the lesson Chroma collection."""
    # TODO: Step 5 - create a PersistentClient and get or reset the collection.
    raise NotImplementedError("Complete get_collection() in Step 5.")


def seed_collection() -> int:
    """Store the facility documents, embeddings, and metadata in Chroma."""
    # TODO: Step 6 - embed each document and add it to the collection.
    raise NotImplementedError("Complete seed_collection() in Step 6.")


def query_collection(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Retrieve the most relevant Chroma results for a user question."""
    # TODO: Step 7 - embed the question, query Chroma, and normalize the results.
    raise NotImplementedError("Complete query_collection() in Step 7.")
