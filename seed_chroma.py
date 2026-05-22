"""Seed the local Chroma database for the Flask RAG API lesson."""

from chroma_store import seed_collection


def main() -> None:
    """Create a fresh Chroma collection and store the lesson documents."""
    count = seed_collection()
    print(f"Seeded {count} facility documents into Chroma.")


if __name__ == "__main__":
    main()
