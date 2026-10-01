"""Ollama helpers for the Flask RAG API technical lesson."""

from typing import List

import ollama


EMBEDDING_MODEL = "embeddinggemma"
GENERATION_MODEL = "llama3.2"


class ModelServiceError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


# def get_embedding(text: str) -> List[float]:
#     """Return one embedding vector for one text input."""
#     # TODO: Step 3 - call ollama.embed(...) and return the first embedding.
#     raise NotImplementedError("Complete get_embedding() in Step 3.")

def get_embedding(text: str) -> List[float]:
    """Return one embedding vector for one text input."""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        return response["embeddings"][0]
    except Exception as exc:
        raise ModelServiceError(
            "The local AI model service is unavailable. "
            "Confirm Ollama is running and the embedding model has been pulled."
        ) from exc


def generate_answer(prompt: str) -> str:
    """Return one generated answer for one prompt."""
    try:
        response = ollama.generate(model=GENERATION_MODEL, prompt=prompt)
        return response["response"].strip()
    except Exception as exc:
        raise ModelServiceError(
            "The local AI model service is unavailable. "
            "Confirm Ollama is running and the generation model has been pulled."
        ) from exc