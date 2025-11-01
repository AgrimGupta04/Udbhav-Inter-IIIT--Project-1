from typing import List
from ml_rag.pdf_parser import extract_text_from_pdf, extract_from_multiple_pdfs
from ml_rag.generator import generate_summary_and_diagnosis
from ml_rag.embedder import get_embedding

def process_reports(pdf_files: List[bytes]):
    """
    Main pipeline: extract text, embed, and call LLM for summary + diagnosis.
    Handles empty or invalid files gracefully.
    """
    combined_text = ""

    # -----------------------------
    # 1. Extract text
    # -----------------------------
    try:
        if not pdf_files:
            raise ValueError("No PDF files provided.")

        if len(pdf_files) == 1:
            combined_text = extract_text_from_pdf(pdf_files[0])
        else:
            combined_text = extract_from_multiple_pdfs(pdf_files)

        if not combined_text.strip():
            raise ValueError("No readable text extracted from the uploaded PDF(s).")

    except Exception as e:
        return {
            "summary_and_diagnosis": {
                "raw_response": f"⚠️ Error during PDF text extraction: {e}",
                "parsed_response": None,
            },
            "embedding_vector_dim": 0,
        }

    # -----------------------------
    # 2. Embed text
    # -----------------------------
    try:
        embedding = get_embedding(combined_text)
        embedding_dim = len(embedding)
    except Exception:
        embedding = []
        embedding_dim = 0

    # -----------------------------
    # 3. Generate summary + diagnosis
    # -----------------------------
    try:
        result = generate_summary_and_diagnosis(combined_text)
    except Exception as e:
        result = {
            "raw_response": f"⚠️ LLM call failed: {e}",
            "parsed_response": None,
        }

    return {
        "summary_and_diagnosis": result,
        "embedding_vector_dim": embedding_dim,
    }