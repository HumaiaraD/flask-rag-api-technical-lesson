"""Ollama helpers for the Flask RAG API technical lesson."""

from typing import List

import ollama


EMBEDDING_MODEL = "embeddinggemma"
GENERATION_MODEL = "llama3.2"


class ModelServiceError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


def get_embedding(text: str) -> List[float]:
    """Return one embedding vector for one text input."""
    # TODO: Step 3 - call ollama.embed(...) and return the first embedding.
    raise NotImplementedError("Complete get_embedding() in Step 3.")


def generate_answer(prompt: str) -> str:
    """Return one generated answer for one prompt."""
    # TODO: Step 4 - call ollama.generate(...) and return the response text.
    raise NotImplementedError("Complete generate_answer() in Step 4.")
