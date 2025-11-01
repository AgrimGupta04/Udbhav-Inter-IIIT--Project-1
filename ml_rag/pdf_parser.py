import io
from PyPDF2 import PdfReader
from typing import List

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a single PDF file (bytes)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def extract_from_multiple_pdfs(files: List[bytes]) -> str:
    """Combine text from multiple PDFs into one string."""
    combined_text = ""
    for f in files:
        combined_text += extract_text_from_pdf(f) + "\n\n"
    return combined_text.strip()
