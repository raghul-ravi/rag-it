"""
Document ingestion pipeline.

This module orchestrates the entire document processing flow:
    1. Find documents in the documents directory
    2. Parse each document to extract text
    3. Split text into chunks
    4. Generate embeddings for each chunk
    5. Store chunks and embeddings in the vector database

The pipeline flow:
    Documents → Parse → Chunk → Embed → Store

Usage:
    # From command line
    python -m src.ingest

    # From code
    from src.ingest import ingest_all_documents
    ingest_all_documents()
"""

from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS,
)
from src.embeddings import Embedder
from src.parsers import parse_document
from src.vectorstore import VectorStore


def get_documents_to_ingest(documents_dir: Path = DOCUMENTS_DIR) -> List[Path]:
    """
    Find all supported documents in the documents directory.

    Args:
        documents_dir: Directory to scan for documents

    Returns:
        List of Path objects for each supported document
    """
    documents = []

    # Ensure directory exists
    if not documents_dir.exists():
        print(f"Creating documents directory: {documents_dir}")
        documents_dir.mkdir(parents=True, exist_ok=True)
        return documents

    # Find all files with supported extensions
    for ext in SUPPORTED_EXTENSIONS.keys():
        # Use glob to find files (recursive with **)
        pattern = f"**/*{ext}"
        found = list(documents_dir.glob(pattern))
        documents.extend(found)

    # Sort by name for consistent ordering
    documents.sort(key=lambda p: p.name)

    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter which tries to split on:
    1. Paragraphs (double newlines)
    2. Sentences (periods, etc.)
    3. Words (spaces)
    4. Characters (last resort)

    Args:
        text: The full text to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters to overlap between chunks

    Returns:
        List of text chunks

    Example:
        >>> chunks = chunk_text("Long document text...", chunk_size=500)
        >>> len(chunks)
        10
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Try these in order
    )

    chunks = splitter.split_text(text)
    return chunks


def ingest_document(
    file_path: Path,
    embedder: Embedder,
    vector_store: VectorStore,
) -> int:
    """
    Ingest a single document into the vector store.

    Args:
        file_path: Path to the document
        embedder: Embedder instance for generating embeddings
        vector_store: VectorStore instance for storage

    Returns:
        Number of chunks added
    """
    print(f"\n📄 Processing: {file_path.name}")

    # Step 1: Parse document to extract text
    try:
        text = parse_document(file_path)
        print(f"   Extracted {len(text)} characters")
    except Exception as e:
        print(f"   ❌ Failed to parse: {e}")
        return 0

    if not text.strip():
        print(f"   ⚠️  No text extracted (file may be empty or scanned)")
        return 0

    # Step 2: Split into chunks
    chunks = chunk_text(text)
    print(f"   Split into {len(chunks)} chunks")

    if not chunks:
        return 0

    # Step 3: Generate embeddings for all chunks
    print(f"   Generating embeddings...")
    embeddings = embedder.embed_texts(chunks)

    # Step 4: Prepare metadata and IDs
    source = str(file_path)
    metadatas = [
        {
            "source": source,
            "filename": file_path.name,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    # Create unique IDs: filename_chunkindex
    # This allows us to update specific chunks later
    ids = [f"{file_path.stem}_{i}" for i in range(len(chunks))]

    # Step 5: Delete existing documents from this source (if re-ingesting)
    vector_store.delete_by_source(source)

    # Step 6: Store in vector database
    vector_store.add_documents(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"   ✅ Added {len(chunks)} chunks to vector store")
    return len(chunks)


def ingest_all_documents(documents_dir: Path = DOCUMENTS_DIR) -> dict:
    """
    Ingest all documents from the documents directory.

    This is the main entry point for bulk ingestion.

    Args:
        documents_dir: Directory containing documents to ingest

    Returns:
        Dictionary with ingestion statistics
    """
    print("=" * 60)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    # Find documents
    documents = get_documents_to_ingest(documents_dir)

    if not documents:
        print(f"\n⚠️  No documents found in: {documents_dir}")
        print(f"   Supported formats: {list(SUPPORTED_EXTENSIONS.keys())}")
        return {"documents_found": 0, "documents_processed": 0, "total_chunks": 0}

    print(f"\n📁 Found {len(documents)} documents to process:")
    for doc in documents:
        print(f"   - {doc.name}")

    # Initialize embedder and vector store
    print("\n🔧 Initializing components...")
    embedder = Embedder()
    vector_store = VectorStore()

    # Process each document
    total_chunks = 0
    successful = 0

    for doc_path in documents:
        chunks_added = ingest_document(doc_path, embedder, vector_store)
        if chunks_added > 0:
            successful += 1
            total_chunks += chunks_added

    # Print summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Documents found:     {len(documents)}")
    print(f"Documents processed: {successful}")
    print(f"Total chunks stored: {total_chunks}")

    # Print vector store stats
    stats = vector_store.get_stats()
    print(f"\nVector store now contains:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   From {stats['num_sources']} source files")

    return {
        "documents_found": len(documents),
        "documents_processed": successful,
        "total_chunks": total_chunks,
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    # Check for --clear flag to reset the database
    if "--clear" in sys.argv:
        print("Clearing vector store...")
        store = VectorStore()
        store.clear_all()
        print("Done! Run without --clear to ingest documents.")
    else:
        ingest_all_documents()
