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
        payload = request.get_json(silent=True) or {}
        question = payload.get("question")

        if not isinstance(question, str) or not question.strip():
            return jsonify({"error": "Question is required."}), 400

        try:
            response = answer_question(question.strip())
        except ModelServiceError as exc:
            return jsonify({"error": str(exc)}), 503

        return jsonify(response), 200

    
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
