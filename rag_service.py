"""RAG service helpers for retrieval, prompt construction, and answer generation."""

from typing import Any, Dict, List

from ai_client import generate_answer
from chroma_store import query_collection


MIN_RELEVANCE_SCORE = 0.25
FALLBACK_ANSWER = "I do not have enough approved facilities context to answer that question."


def has_usable_context(results: List[Dict[str, Any]]) -> bool:
    """Return True when retrieval produced context strong enough to use."""
    if not results:
        return False

    top_score = float(results[0].get("score", 0.0))
    return top_score >= MIN_RELEVANCE_SCORE

def format_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved results as labeled context for the prompt."""
    context_blocks = []

    for index, result in enumerate(results, start=1):
        context_blocks.append(
            "\\n".join(
                [
                    f"Source {index}",
                    f"ID: {result['id']}",
                    f"Title: {result['title']}",
                    f"Category: {result['category']}",
                    f"Text: {result['text']}",
                ]
            )
        )

    return "\\n\\n".join(context_blocks)

def build_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return source metadata for the API response."""
    return [
        {
            "id": result["id"],
            "title": result["title"],
            "category": result["category"],
            "score": round(float(result["score"]), 4),
        }
        for result in results
    ]



def build_prompt(question: str, results: List[Dict[str, Any]]) -> str:
    """Build a structured prompt from the user question and retrieved context."""
    context = format_context(results)

    return f"""

You are a facilities operations assistant for employees.

Use only the approved facilities context below to answer the user's question.

If the context does not contain enough information, say:

"I do not have enough approved facilities context to answer that question."


Approved facilities context:
{context}

User question:
{question}

Response requirements:

- Answer in 2 to 4 sentences.

- Use a helpful and professional tone.

- Do not invent policies, form names, phone numbers, timelines, or approval steps.

- Base the answer only on the approved context.

""".strip()


def answer_question(question: str) -> Dict[str, Any]:
    """Run retrieval, prompt construction, generation, and source formatting."""
    results = query_collection(question)

    if not has_usable_context(results):
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

    prompt = build_prompt(question, results)
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": build_sources(results),
    }

