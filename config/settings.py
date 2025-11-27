"""
Configuration settings for the Personal RAG system.

This file centralizes all configurable values. Modify these instead of
hardcoding values throughout the codebase.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# PATHS
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Where users place their documents to be ingested
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

# Where ChromaDB stores the vector database
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

# =============================================================================
# CHUNKING SETTINGS
# =============================================================================

# How many characters per chunk (roughly 100-150 words)
# Smaller = more precise retrieval, but less context
# Larger = more context, but may retrieve irrelevant info
CHUNK_SIZE = 500

# Overlap between chunks to avoid cutting sentences
# This ensures context isn't lost at chunk boundaries
CHUNK_OVERLAP = 50

# =============================================================================
# EMBEDDING SETTINGS
# =============================================================================

# The model used to convert text into vectors
# Options:
#   - "all-MiniLM-L6-v2" (fast, 384 dimensions, good for most cases)
#   - "all-mpnet-base-v2" (slower, 768 dimensions, more accurate)
#   - "multi-qa-MiniLM-L6-cos-v1" (optimized for Q&A)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# =============================================================================
# VECTOR STORE SETTINGS
# =============================================================================

# Name of the ChromaDB collection
COLLECTION_NAME = "personal_documents"

# How many similar chunks to retrieve for each query
# Higher = more context but may include less relevant results
TOP_K = 5

# =============================================================================
# LLM SETTINGS
# =============================================================================

# Which LLM provider to use: "ollama" (local) or "openai" (cloud)
LLM_PROVIDER = "ollama"

# Ollama settings (local LLM)
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# OpenAI settings (cloud LLM)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# =============================================================================
# SUPPORTED FILE TYPES
# =============================================================================

# Map file extensions to parser types
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",  # Treat .doc as .docx (may not work for old .doc files)
    ".txt": "text",
    ".md": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".bmp": "image",
}
