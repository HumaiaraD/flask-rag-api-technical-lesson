"""Flask app for the local Chroma + Ollama RAG API lesson."""

from flask import Flask, jsonify, request

from ai_client import ModelServiceError
from rag_service import answer_question


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    @app.post("/api/ask")
    def ask():
        """Accept a question and return a source-backed RAG response."""
        # TODO: Step 11 - validate the request, call answer_question(), and return JSON.
        return jsonify({"message": "Complete /api/ask in Step 11."}), 501

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
