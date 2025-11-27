"""
Plain text file parser.

Handles .txt and .md (Markdown) files. Simply reads the file
content as UTF-8 text.

Note:
    - Assumes UTF-8 encoding
    - Falls back to latin-1 if UTF-8 fails
"""

from pathlib import Path
from typing import List, Union


def parse_text(file_path: Union[str, Path]) -> str:
    """
    Read text from a plain text file.

    Args:
        file_path: Path to the text file (.txt, .md, etc.)

    Returns:
        File content as string

    Example:
        >>> text = parse_text("notes.txt")
        >>> print(text[:100])
    """
    file_path = Path(file_path)

    # Try UTF-8 first, fall back to latin-1
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read().strip()


# Test the parser directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        text_path = sys.argv[1]
        text = parse_text(text_path)
        print(f"Read {len(text)} characters from {text_path}")
        print("\n--- First 500 characters ---")
        print(text[:500])
    else:
        print("Usage: python -m src.parsers.text <path_to_text_file>")
