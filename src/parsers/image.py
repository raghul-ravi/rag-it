"""
Image parser using OCR (Optical Character Recognition).

Uses pytesseract (Python wrapper for Tesseract OCR) to extract
text from images. Useful for scanned documents, screenshots,
or photos of documents.

Prerequisites:
    Tesseract must be installed on your system:
    - macOS: brew install tesseract
    - Ubuntu: sudo apt-get install tesseract-ocr
    - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

Note:
    - Quality depends heavily on image quality
    - Works best with clear, high-contrast text
    - Handwritten text has poor accuracy
"""

from pathlib import Path
from typing import Union

import pytesseract
from PIL import Image

# Set tesseract path for macOS (Homebrew installation)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


def parse_image(file_path: Union[str, Path]) -> str:
    """
    Extract text from an image using OCR.

    Args:
        file_path: Path to the image file (PNG, JPG, TIFF, BMP)

    Returns:
        Extracted text from the image

    Raises:
        pytesseract.TesseractNotFoundError: If Tesseract is not installed

    Example:
        >>> text = parse_image("scanned_transcript.png")
        >>> print(text[:100])
        "Academic Transcript\\nName: John Doe\\n..."
    """
    file_path = Path(file_path)

    # Open image with Pillow
    image = Image.open(file_path)

    # Run OCR
    # config options:
    #   --psm 1: Automatic page segmentation with OSD
    #   --psm 3: Fully automatic page segmentation (default)
    #   --psm 6: Assume a single uniform block of text
    text = pytesseract.image_to_string(image, config="--psm 3")

    return text.strip()


def check_tesseract_installed() -> bool:
    """Check if Tesseract OCR is installed and accessible."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False


# Test the parser directly
if __name__ == "__main__":
    import sys

    # First check if Tesseract is installed
    if not check_tesseract_installed():
        print("ERROR: Tesseract is not installed!")
        print("\nInstall it with:")
        print("  macOS:  brew install tesseract")
        print("  Ubuntu: sudo apt-get install tesseract-ocr")
        print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        text = parse_image(image_path)
        print(f"Extracted {len(text)} characters from {image_path}")
        print("\n--- Extracted text ---")
        print(text)
    else:
        print("Tesseract is installed and working!")
        print("\nUsage: python -m src.parsers.image <path_to_image>")
