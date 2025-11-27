"""
RAG-It Chat UI using Gradio.

A modern web interface for querying your documents.

Usage:
    python -m src.app

Then open http://localhost:7860 in your browser.
"""

import os

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
from src.query import QueryEngine

# Initialize RAG engine once at startup
print("Initializing RAG engine...")
engine = QueryEngine()


def respond(message: str, history: list) -> str:
    """
    Process a user message and return the RAG response.

    Args:
        message: The user's question
        history: Chat history (not used for RAG, but required by Gradio)

    Returns:
        The answer from the RAG system
    """
    if not message.strip():
        return "Please enter a question."

    try:
        answer = engine.query(message, show_sources=True)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"


def get_stats() -> str:
    """Get vector store statistics."""
    stats = engine.vector_store.get_stats()
    return f"**Documents loaded:** {stats['total_documents']} chunks from {stats['num_sources']} files"


# Example questions for users to try
EXAMPLES = [
    "Where did I study?",
    "What degree did I receive?",
    "When did I graduate?",
    "What were my grades in semester 2?",
    "What was my register number?",
]

# Create the Gradio interface using ChatInterface (Gradio 6.x API)
demo = gr.ChatInterface(
    fn=respond,
    title="RAG-It",
    description="Just RAG it! Ask questions about your documents",
    examples=EXAMPLES,
)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("RAG-It Chat UI (Gradio)")
    print("=" * 50)
    print("\nOpen http://localhost:7860 in your browser")
    print("Press Ctrl+C to stop the server\n")

    demo.launch(server_name="0.0.0.0", server_port=7860)
