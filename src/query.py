"""
Query module for the RAG system.

This module handles:
    1. Converting user questions into embeddings
    2. Finding relevant document chunks via similarity search
    3. Building a prompt with context
    4. Getting an answer from the LLM

The RAG flow:
    Question → Embed → Search → Context → LLM → Answer

Usage:
    # Single query
    python -m src.query "Where did I study?"

    # Interactive mode
    python -m src.query --interactive
"""

from typing import List

import ollama
from openai import OpenAI

from config.settings import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K,
)
from src.embeddings import Embedder
from src.vectorstore import VectorStore


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

# This prompt tells the LLM how to behave
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. ONLY use information from the provided context to answer
2. If the context doesn't contain enough information, say "I don't have enough information to answer that based on your documents."
3. Be specific and quote from the context when relevant
4. If asked about dates, grades, or specific facts, provide exact values from the context
5. Keep answers concise but complete"""

CONTEXT_TEMPLATE = """Context from your documents:
---
{context}
---

Question: {question}

Answer based on the context above:"""


# =============================================================================
# LLM CLIENTS
# =============================================================================

def get_ollama_response(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """
    Get a response from Ollama (local LLM).

    Args:
        prompt: The user prompt with context
        system_prompt: Instructions for the LLM

    Returns:
        The LLM's response text
    """
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


def get_openai_response(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """
    Get a response from OpenAI API.

    Args:
        prompt: The user prompt with context
        system_prompt: Instructions for the LLM

    Returns:
        The LLM's response text
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


def get_llm_response(prompt: str) -> str:
    """
    Get a response from the configured LLM provider.

    Uses LLM_PROVIDER from settings to determine which service to use.
    """
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
        return get_openai_response(prompt)
    else:
        return get_ollama_response(prompt)


# =============================================================================
# QUERY ENGINE
# =============================================================================

class QueryEngine:
    """
    Main query engine that orchestrates the RAG pipeline.

    Usage:
        engine = QueryEngine()
        answer = engine.query("Where did I study?")
        print(answer)
    """

    def __init__(self):
        """Initialize the query engine with embedder and vector store."""
        print("Initializing query engine...")
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        print(f"Ready! Using {LLM_PROVIDER} as LLM provider.\n")

    def retrieve_context(self, question: str, top_k: int = TOP_K) -> List[dict]:
        """
        Retrieve relevant document chunks for the question.

        Args:
            question: The user's question
            top_k: Number of chunks to retrieve

        Returns:
            List of dicts with 'text', 'source', and 'distance' keys
        """
        # Embed the question
        query_embedding = self.embedder.embed_text(question)

        # Search vector store
        results = self.vector_store.query(query_embedding, top_k=top_k)

        # Format results
        contexts = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                contexts.append({
                    "text": doc,
                    "source": meta.get("filename", "unknown"),
                    "distance": dist,
                })

        return contexts

    def build_prompt(self, question: str, contexts: List[dict]) -> str:
        """
        Build the prompt with context for the LLM.

        Args:
            question: The user's question
            contexts: Retrieved document chunks

        Returns:
            Formatted prompt string
        """
        # Format context with source attribution
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            context_parts.append(f"[{i}] From {ctx['source']}:\n{ctx['text']}")

        context_str = "\n\n".join(context_parts)

        # Fill in the template
        prompt = CONTEXT_TEMPLATE.format(
            context=context_str,
            question=question,
        )

        return prompt

    def query(self, question: str, top_k: int = TOP_K, show_sources: bool = True) -> str:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: The user's question
            top_k: Number of context chunks to retrieve
            show_sources: Whether to show source files in the response

        Returns:
            The answer string
        """
        # Step 1: Retrieve relevant context
        contexts = self.retrieve_context(question, top_k=top_k)

        if not contexts:
            return "I couldn't find any relevant information in your documents. Make sure you've ingested some documents first."

        # Step 2: Build prompt with context
        prompt = self.build_prompt(question, contexts)

        # Step 3: Get LLM response
        try:
            answer = get_llm_response(prompt)
        except Exception as e:
            return f"Error getting LLM response: {e}\n\nMake sure Ollama is running (ollama serve) or your OpenAI key is set."

        # Step 4: Optionally append sources
        if show_sources:
            sources = set(ctx["source"] for ctx in contexts)
            answer += f"\n\nSources: {', '.join(sources)}"

        return answer


# =============================================================================
# CLI INTERFACE
# =============================================================================

def interactive_mode():
    """Run the query engine in interactive mode."""
    print("=" * 60)
    print("RAG-It - Interactive Query Mode")
    print("=" * 60)
    print("Type your questions and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'sources' to see what documents are loaded.")
    print("=" * 60)

    engine = QueryEngine()

    while True:
        try:
            question = input("\nYour question: ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if question.lower() == "sources":
                stats = engine.vector_store.get_stats()
                print(f"\nLoaded documents ({stats['total_documents']} chunks):")
                for source in stats["sources"]:
                    print(f"   - {source}")
                continue

            print("\nSearching documents...")
            answer = engine.query(question)
            print(f"\nAnswer:\n{answer}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def single_query(question: str):
    """Run a single query and print the result."""
    engine = QueryEngine()
    print(f"\nQuestion: {question}\n")
    print("Searching documents...")
    answer = engine.query(question)
    print(f"\nAnswer:\n{answer}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    if "--interactive" in sys.argv or "-i" in sys.argv:
        interactive_mode()
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Join all non-flag arguments as the question
        question = " ".join(arg for arg in sys.argv[1:] if not arg.startswith("-"))
        single_query(question)
    else:
        print("RAG-It Query System")
        print("\nUsage:")
        print('  python -m src.query "Your question here"')
        print("  python -m src.query --interactive")
        print("\nExamples:")
        print('  python -m src.query "Where did I study?"')
        print('  python -m src.query "What certifications do I have?"')
