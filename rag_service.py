"""RAG service helpers for retrieval, prompt construction, and answer generation."""

from typing import Any, Dict, List

from ai_client import generate_answer
from chroma_store import query_collection


MIN_RELEVANCE_SCORE = 0.25
FALLBACK_ANSWER = "I do not have enough approved facilities context to answer that question."


def has_usable_context(results: List[Dict[str, Any]]) -> bool:
    """Return True when retrieval produced context strong enough to use."""
    # TODO: Step 8 - check whether the top result exists and meets the relevance threshold.
    raise NotImplementedError("Complete has_usable_context() in Step 8.")


def format_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved results as labeled context for the prompt."""
    # TODO: Step 8 - format each result with source metadata and source text.
    raise NotImplementedError("Complete format_context() in Step 8.")


def build_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return source metadata for the API response."""
    # TODO: Step 8 - return user-readable source metadata without raw embeddings.
    raise NotImplementedError("Complete build_sources() in Step 8.")


def build_prompt(question: str, results: List[Dict[str, Any]]) -> str:
    """Build a structured prompt from the user question and retrieved context."""
    # TODO: Step 9 - combine instructions, labeled context, user question, and response rules.
    raise NotImplementedError("Complete build_prompt() in Step 9.")


def answer_question(question: str) -> Dict[str, Any]:
    """Run retrieval, prompt construction, generation, and source formatting."""
    # TODO: Step 10 - coordinate the RAG workflow.
    raise NotImplementedError("Complete answer_question() in Step 10.")
