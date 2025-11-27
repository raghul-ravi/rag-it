"""
Embedding generation module.

This module converts text into numerical vectors (embeddings) using
sentence-transformers. These vectors capture the semantic meaning of text,
allowing us to find similar content through vector similarity search.

How it works:
    1. Text goes in: "I studied mathematics at university"
    2. Vector comes out: [0.12, -0.34, 0.56, ...] (384 numbers for MiniLM)
    3. Similar text = similar vectors (measured by cosine similarity)

Example:
    >>> embedder = Embedder()
    >>> vector = embedder.embed_text("Hello world")
    >>> print(len(vector))  # 384 for all-MiniLM-L6-v2
"""

from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL


class Embedder:
    """
    Wrapper around sentence-transformers for generating text embeddings.

    The model is loaded once when the class is instantiated and reused
    for all subsequent embedding requests.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is from settings.py

        Note:
            First run will download the model (~90MB for MiniLM).
            Subsequent runs use the cached version.
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Model loaded. Embedding dimension: {self.get_dimension()}")

    def embed_text(self, text: str) -> list[float]:
        """
        Convert a single text string into a vector.

        Args:
            text: The text to embed (e.g., "I studied at MIT")

        Returns:
            A list of floats representing the embedding vector.
            Length depends on the model (384 for MiniLM, 768 for mpnet).
        """
        # encode() returns a numpy array, we convert to list for JSON compatibility
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Convert multiple texts into vectors (batch processing).

        This is more efficient than calling embed_text() in a loop
        because the model processes all texts in a single batch.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one per input text.
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Integer dimension (e.g., 384 for MiniLM, 768 for mpnet)
        """
        return self.model.get_sentence_embedding_dimension()


# =============================================================================
# USAGE EXAMPLE (run this file directly to test)
# =============================================================================

if __name__ == "__main__":
    # Create embedder instance
    embedder = Embedder()

    # Test with sample texts
    texts = [
        "I studied computer science at Stanford University",
        "My major was CS at Stanford",
        "I love eating pizza on weekends",
    ]

    print("\nGenerating embeddings for sample texts...")
    embeddings = embedder.embed_texts(texts)

    # Show results
    for text, emb in zip(texts, embeddings):
        print(f"\nText: {text}")
        print(f"Vector (first 5 dims): {emb[:5]}")

    # Demonstrate similarity (texts 0 and 1 should be more similar than 0 and 2)
    import numpy as np
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    sim_01 = cosine_similarity(embeddings[0], embeddings[1])
    sim_02 = cosine_similarity(embeddings[0], embeddings[2])

    print(f"\nSimilarity between text 0 and 1 (both about Stanford CS): {sim_01:.4f}")
    print(f"Similarity between text 0 and 2 (Stanford vs pizza): {sim_02:.4f}")
    print("(Higher = more similar, range is -1 to 1)")
