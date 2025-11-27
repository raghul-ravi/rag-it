"""
PDF document parser.

Uses pypdf to extract text from PDF files. For scanned PDFs (images),
falls back to OCR using pdf2image and pytesseract.

Note:
    - Works well with text-based PDFs
    - Scanned PDFs use OCR (requires Tesseract and poppler)
    - Some PDFs with complex layouts may have jumbled text
"""

import io
from pathlib import Path
from typing import Union

from pypdf import PdfReader
from PIL import Image
import pytesseract

# Set tesseract path for macOS (Homebrew installation)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


def pdf_page_to_image(page, page_num: int) -> Image.Image:
    """Convert a PDF page to an image for OCR."""
    # Try to extract images from the page
    if "/XObject" in page["/Resources"]:
        x_objects = page["/Resources"]["/XObject"].get_object()
        for obj_name in x_objects:
            obj = x_objects[obj_name]
            if obj["/Subtype"] == "/Image":
                # Extract the image
                if "/Filter" in obj:
                    if obj["/Filter"] == "/DCTDecode":
                        # JPEG
                        data = obj._data
                        return Image.open(io.BytesIO(data))
                    elif obj["/Filter"] == "/FlateDecode":
                        # PNG-like
                        data = obj.get_data()
                        width = obj["/Width"]
                        height = obj["/Height"]
                        if "/ColorSpace" in obj:
                            mode = "RGB" if obj["/ColorSpace"] == "/DeviceRGB" else "L"
                        else:
                            mode = "RGB"
                        try:
                            return Image.frombytes(mode, (width, height), data)
                        except Exception:
                            pass
    return None


def parse_pdf(file_path: Union[str, Path]) -> str:
    """
    Extract text from a PDF file.

    For text-based PDFs, extracts text directly.
    For scanned PDFs (images), uses OCR.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text from all pages, joined by newlines

    Example:
        >>> text = parse_pdf("transcript.pdf")
        >>> print(text[:100])
        "Academic Transcript\\nName: John Doe\\n..."
    """
    file_path = Path(file_path)
    reader = PdfReader(file_path)

    pages_text = []

    for page_num, page in enumerate(reader.pages, start=1):
        # First try direct text extraction
        text = page.extract_text()

        if text and text.strip():
            pages_text.append(f"[Page {page_num}]\n{text}")
        else:
            # No text found - try OCR on embedded images
            print(f"   Page {page_num}: No text found, trying OCR...")
            image = pdf_page_to_image(page, page_num)
            if image:
                try:
                    ocr_text = pytesseract.image_to_string(image, config="--psm 3")
                    if ocr_text.strip():
                        pages_text.append(f"[Page {page_num}]\n{ocr_text}")
                except Exception as e:
                    print(f"   OCR failed for page {page_num}: {e}")

    full_text = "\n\n".join(pages_text)
    return full_text.strip()


# Test the parser directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        text = parse_pdf(pdf_path)
        print(f"Extracted {len(text)} characters from {pdf_path}")
        print("\n--- First 500 characters ---")
        print(text[:500])
    else:
        print("Usage: python -m src.parsers.pdf <path_to_pdf>")
