# CLAUDE.md

This file provides context for AI assistants working on this codebase.

## Project Overview

**RAG-It** - A personal RAG (Retrieval-Augmented Generation) system for querying personal documents using natural language. The system ingests documents, converts them to vector embeddings, stores them in ChromaDB, and answers questions using retrieved context + an LLM.

## Tech Stack

- **Python 3.9+**
- **ChromaDB**: Vector database (local, persistent storage in `data/chroma_db/`)
- **sentence-transformers**: Embedding generation (default: all-MiniLM-L6-v2)
- **Ollama/OpenAI**: LLM for generating answers
- **Document parsers**: pypdf, python-docx, pytesseract, unstructured

## Architecture

```
Documents → Parsers → Chunking → Embeddings → ChromaDB
                                                  ↓
User Query → Embedding → Similarity Search → Context → LLM → Answer
```

## Key Files

- `src/ingest.py` - Main ingestion pipeline, orchestrates parsing and storage
- `src/embeddings.py` - Wrapper around sentence-transformers
- `src/vectorstore.py` - ChromaDB operations (add, query, delete)
- `src/query.py` - Query processing, context retrieval, LLM interaction
- `src/parsers/` - Document type parsers (pdf, docx, image, text)
- `config/settings.py` - All configuration constants

## Common Tasks

### Adding a new document parser
1. Create `src/parsers/<type>.py` with a `parse(file_path: str) -> str` function
2. Add to `PARSER_MAP` in `src/parsers/__init__.py`

### Changing the embedding model
1. Update `EMBEDDING_MODEL` in `config/settings.py`
2. Delete `data/chroma_db/` (embeddings are model-specific)
3. Re-run ingestion

### Switching LLM provider
1. Set `LLM_PROVIDER` in `config/settings.py` to "ollama" or "openai"
2. If OpenAI, ensure `OPENAI_API_KEY` is set in `.env`

## Code Conventions

- Type hints on all function signatures
- Docstrings for public functions
- Config values in `config/settings.py`, not hardcoded
- Parsers return plain text strings
- ChromaDB collection name: "personal_documents"

## Testing

```bash
pytest tests/
pytest tests/test_ingest.py -v  # Specific file
```

## Common Issues

- **Dimension mismatch**: Happens when switching embedding models without clearing the DB
- **Import errors**: Ensure virtual environment is activated
- **OCR failures**: Check Tesseract installation with `tesseract --version`

## Environment Variables

- `OPENAI_API_KEY` - Required only if using OpenAI as LLM provider

## Data Storage

- `data/documents/` - Source documents (user-provided)
- `data/chroma_db/` - Vector database (auto-generated, can be deleted to reset)
