"""
Vector store module using ChromaDB.

ChromaDB is a vector database that stores embeddings and allows
similarity search. Think of it as a database optimized for finding
"similar" items rather than exact matches.

How it works:
    1. Store: Document chunk + its embedding + metadata
    2. Query: Your question embedding → find similar chunk embeddings
    3. Return: The original text chunks that match your query

Key concepts:
    - Collection: Like a table in a regular database
    - Document: The original text chunk
    - Embedding: The vector representation of the text
    - Metadata: Extra info (source file, page number, etc.)
"""

from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from config.settings import CHROMA_DB_DIR, COLLECTION_NAME, TOP_K


class VectorStore:
    """
    Wrapper around ChromaDB for storing and querying document embeddings.

    Usage:
        store = VectorStore()
        store.add_documents(texts, embeddings, metadatas)
        results = store.query(query_embedding, top_k=5)
    """

    def __init__(self, persist_directory: Path = CHROMA_DB_DIR):
        """
        Initialize the vector store.

        Args:
            persist_directory: Where to store the database files.
                              Data persists between runs.
        """
        # Ensure directory exists
        persist_directory = Path(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        # Settings explanation:
        #   - persist_directory: Where to save the database
        #   - anonymized_telemetry: Disable usage tracking
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create the collection
        # If collection exists, it's loaded; otherwise, created
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Personal documents for RAG"}
        )

        print(f"Vector store initialized at: {persist_directory}")
        print(f"Collection '{COLLECTION_NAME}' has {self.collection.count()} documents")

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: List[str],
    ) -> None:
        """
        Add documents with their embeddings to the vector store.

        Args:
            documents: List of text chunks
            embeddings: List of embedding vectors (same length as documents)
            metadatas: List of metadata dicts (source file, chunk index, etc.)
            ids: Unique identifiers for each document

        Example:
            store.add_documents(
                documents=["I studied at MIT", "I majored in CS"],
                embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
                metadatas=[{"source": "resume.pdf"}, {"source": "resume.pdf"}],
                ids=["resume_0", "resume_1"]
            )
        """
        if not documents:
            print("No documents to add")
            return

        # ChromaDB add() method
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        print(f"Added {len(documents)} documents to vector store")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = TOP_K,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Find documents similar to the query embedding.

        Args:
            query_embedding: Vector representation of the query
            top_k: Number of results to return
            where: Optional filter (e.g., {"source": "resume.pdf"})

        Returns:
            Dictionary with keys:
                - documents: List of matching text chunks
                - metadatas: List of metadata for each match
                - distances: List of distances (lower = more similar)
                - ids: List of document IDs

        Example:
            results = store.query(query_embedding, top_k=3)
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                print(f"[{dist:.4f}] {meta['source']}: {doc[:50]}...")
        """
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            query_params["where"] = where

        results = self.collection.query(**query_params)

        return results

    def delete_by_source(self, source: str) -> None:
        """
        Delete all documents from a specific source file.

        Useful when you want to re-ingest a file after updates.

        Args:
            source: The source file path to delete
        """
        # Get IDs of documents with this source
        results = self.collection.get(
            where={"source": source},
            include=[]  # We only need IDs
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            print(f"Deleted {len(results['ids'])} documents from source: {source}")
        else:
            print(f"No documents found for source: {source}")

    def clear_all(self) -> None:
        """
        Delete all documents from the collection.

        WARNING: This is irreversible! You'll need to re-ingest everything.
        """
        # Get all IDs
        all_docs = self.collection.get(include=[])

        if all_docs["ids"]:
            self.collection.delete(ids=all_docs["ids"])
            print(f"Cleared all {len(all_docs['ids'])} documents from vector store")
        else:
            print("Vector store is already empty")

    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with count and sources
        """
        count = self.collection.count()

        # Get unique sources
        if count > 0:
            all_docs = self.collection.get(include=["metadatas"])
            sources = set()
            for meta in all_docs["metadatas"]:
                if meta and "source" in meta:
                    sources.add(meta["source"])
            sources = sorted(sources)
        else:
            sources = []

        return {
            "total_documents": count,
            "sources": sources,
            "num_sources": len(sources),
        }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Initialize store
    store = VectorStore()

    # Print stats
    stats = store.get_stats()
    print(f"\nVector Store Stats:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Number of sources: {stats['num_sources']}")
    if stats['sources']:
        print(f"  Sources: {stats['sources']}")
