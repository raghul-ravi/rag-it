"""
Document parsers package.

This package contains parsers for different file types. Each parser
extracts plain text from its respective file format.

Usage:
    from src.parsers import parse_document

    text = parse_document("/path/to/file.pdf")
"""

from pathlib import Path
from typing import List, Union

from config.settings import SUPPORTED_EXTENSIONS

from .docx import parse_docx
from .image import parse_image
from .pdf import parse_pdf
from .text import parse_text

# Map parser types to their functions
PARSER_MAP = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "image": parse_image,
    "text": parse_text,
}


def parse_document(file_path: Union[str, Path]) -> str:
    """
    Parse any supported document and extract its text.

    This is the main entry point for document parsing. It automatically
    detects the file type and uses the appropriate parser.

    Args:
        file_path: Path to the document file

    Returns:
        Extracted text content as a string

    Raises:
        ValueError: If the file type is not supported
        FileNotFoundError: If the file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get file extension (lowercase)
    extension = file_path.suffix.lower()

    # Check if extension is supported
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    # Get parser type and function
    parser_type = SUPPORTED_EXTENSIONS[extension]
    parser_func = PARSER_MAP[parser_type]

    # Parse and return text
    return parser_func(file_path)


def get_supported_extensions() -> List[str]:
    """Return list of supported file extensions."""
    return list(SUPPORTED_EXTENSIONS.keys())
