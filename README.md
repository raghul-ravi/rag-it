# RAG-It

A local-first Retrieval-Augmented Generation (RAG) system for querying your personal documents using natural language. Just RAG it!

## Features

- **Multi-format document support**: PDF, Word (.docx), text files, images (with OCR)
- **Fully local**: All data stays on your machine
- **Semantic search**: Find information based on meaning, not just keywords
- **LLM integration**: Get natural language answers from your documents
- **Web UI**: Beautiful chat interface powered by Gradio

## Tech Stack

- **Language**: Python 3.9+
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Ollama (local) or OpenAI API
- **Document Parsing**: pypdf, python-docx, pytesseract
- **Web UI**: Gradio

## Project Structure

```
rag-it/
├── src/
│   ├── __init__.py
│   ├── app.py             # Gradio web UI
│   ├── ingest.py          # Document ingestion pipeline
│   ├── embeddings.py      # Embedding generation
│   ├── vectorstore.py     # ChromaDB operations
│   ├── query.py           # Query processing and LLM interaction
│   └── parsers/
│       ├── __init__.py
│       ├── pdf.py
│       ├── docx.py
│       ├── image.py
│       └── text.py
├── data/
│   ├── documents/         # Place your documents here
│   └── chroma_db/         # Vector database storage
├── config/
│   └── settings.py        # Configuration settings
├── tests/
│   └── ...
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd rag-it
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract (for OCR support)**
   ```bash
   # macOS
   brew install tesseract

   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr

   # Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
   ```

5. **Install Ollama (for local LLM)**
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.2
   ```

## Usage

### 1. Add Documents

Place your documents in the `data/documents/` directory:
```bash
cp ~/Downloads/transcript.pdf data/documents/
cp ~/Documents/resume.docx data/documents/
```

### 2. Ingest Documents

```bash
python -m src.ingest
```

This will:
- Parse all documents in `data/documents/`
- Chunk them into smaller pieces
- Generate embeddings
- Store in ChromaDB

### 3. Query Your Documents

**Web UI (recommended):**
```bash
python -m src.app
```
Then open http://localhost:7860 in your browser.

**Command line:**
```bash
python -m src.query "Where did I study?"
```

**Interactive mode:**
```bash
python -m src.query --interactive
```

## Configuration

Edit `config/settings.py` to customize:

```python
# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunk settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# LLM settings
LLM_PROVIDER = "ollama"  # or "openai"
OLLAMA_MODEL = "llama3.2"
OPENAI_MODEL = "gpt-4o-mini"

# Number of chunks to retrieve
TOP_K = 5
```

## Environment Variables

Create a `.env` file for API keys (if using OpenAI):

```
OPENAI_API_KEY=sk-...
```

## Example Queries

- "Where did I study and when did I graduate?"
- "What was my mathematics mark in high school?"
- "List all my work experience"
- "What certifications do I have?"
- "Summarize my educational background"

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Document Parsers

1. Create a new parser in `src/parsers/`
2. Implement the `parse(file_path) -> str` function
3. Register it in `src/parsers/__init__.py`

## Troubleshooting

### OCR not working
Ensure Tesseract is installed and in your PATH:
```bash
tesseract --version
```

### Slow ingestion
- Reduce `CHUNK_SIZE` for faster processing
- Use a smaller embedding model
- Process documents in batches

### Out of memory
- Process fewer documents at once
- Use a smaller embedding model
- Increase swap space

## License

MIT
