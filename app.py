import os
import re
import json
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Fix import path for ml_rag
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ml_rag")))
from ml_rag.rag_pipeline import process_reports


def sanitize_text(text: str) -> str:
    """Remove non-ASCII characters (like emojis) before writing to PDF."""
    return re.sub(r"[^\x00-\x7F]+", " ", text)


# -------------------------------------------------------------------------
# Streamlit Config + Custom Neon Theme CSS
# -------------------------------------------------------------------------
st.set_page_config(page_title="Clinical Report Summarizer", layout="wide")

st.markdown("""
<style>
/* Neon Futuristic Theme */
body {
    background: linear-gradient(135deg, #00ff9d 0%, #00b3ff 100%);
    font-family: 'Poppins', sans-serif;
    color: #111;
}

/* Main app container */
.main {
    background-color: rgba(255, 255, 255, 0.92);
    border-radius: 18px;
    padding: 2rem;
    box-shadow: 0 0 25px rgba(0, 255, 200, 0.25);
}

/* Headings */
h1 {
    color: #0d47a1;
    text-align: center;
    font-weight: 800;
    text-shadow: 0px 0px 12px rgba(0, 255, 200, 0.4);
}
h2, h3 {
    color: #004d40;
    font-weight: 700;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00bfa5, #00e676);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    height: 3em;
    width: 100%;
    box-shadow: 0 6px 14px rgba(0, 255, 150, 0.4);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #00e676, #1de9b6);
    box-shadow: 0 8px 18px rgba(0, 255, 200, 0.5);
}

/* Info box styling */
.report-box {
    background-color: #e0f7fa;
    border-left: 6px solid #00bfa5;
    color: #004d40;
    padding: 18px;
    border-radius: 10px;
    margin-top: 10px;
    margin-bottom: 10px;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
}

/* Tabs */
div[data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 10px;
    padding: 6px;
}
div[data-baseweb="tab"] {
    color: #00695c;
    font-weight: 600;
}
div[data-baseweb="tab"]:hover {
    color: #00bfa5;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #00c853 0%, #00bfa5 100%);
    color: white;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] .stButton>button {
    background-color: #69f0ae;
    color: black;
    font-weight: 600;
}

/* Radio & Select */
div[role="radiogroup"] label {
    background: #e0f2f1;
    border-radius: 8px;
    padding: 6px 10px;
    margin-right: 10px;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 1rem;
    color: #004d40;
    font-weight: 700;
    text-shadow: 0px 0px 5px rgba(0, 255, 150, 0.3);
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966485.png", width=80)
    st.title("📂 Upload Reports")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    st.markdown("---")
    st.markdown("💡 **Tip:** Upload multiple reports for combined summarization.")
    st.markdown("---")
    st.markdown("**Developed for Udbhav Hackathon 2025 **")
    st.markdown("<small>Built using Streamlit + OpenAI</small>", unsafe_allow_html=True)


# -------------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------------
if "output_text" not in st.session_state:
    st.session_state.output_text = None
if "embedding_dim" not in st.session_state:
    st.session_state.embedding_dim = 0


# -------------------------------------------------------------------------
# Main Header
# -------------------------------------------------------------------------
st.title("Clinical Report Summarizer 💊")
st.caption("Summarize patient reports and generate prioritized differential diagnoses using LLMs + embeddings.")


# -------------------------------------------------------------------------
# Generate Summary Button
# -------------------------------------------------------------------------
if uploaded_files and st.button("🚀 Generate Summary & Diagnoses"):
    with st.spinner("Analyzing reports... please wait ⏳"):
        pdf_bytes = [f.read() for f in uploaded_files]
        try:
            result = process_reports(pdf_bytes)
            llm_output = result["summary_and_diagnosis"].get("raw_response", "")
            parsed = result["summary_and_diagnosis"].get("parsed_response")
            output_text = ""

            if parsed:
                st.session_state.output_text = ""
                st.success("✅ Analysis complete!")

                summary = parsed.get("summary", "No summary found.")
                output_text += "🧠 Summary:\n" + summary + "\n\n"
                diff_text = "💊 Differential Diagnoses:\n"
                for d in parsed.get("differentials", []):
                    rank = d.get("rank", "?")
                    diag = d.get("diagnosis", "Unknown")
                    rationale = d.get("rationale", "")
                    diff_text += f"{rank}. {diag} — {rationale}\n"

                output_text += diff_text
                st.session_state.output_text = output_text
                st.session_state.embedding_dim = result["embedding_vector_dim"]

                tab1, tab2 = st.tabs(["🧠 Summary", "💊 Differential Diagnoses"])
                with tab1:
                    st.markdown(f"<div class='report-box'>{summary}</div>", unsafe_allow_html=True)
                with tab2:
                    for d in parsed.get("differentials", []):
                        rank = d.get("rank", "?")
                        diag = d.get("diagnosis", "Unknown")
                        rationale = d.get("rationale", "")
                        st.markdown(
                            f"<div class='report-box'><b>{rank}. {diag}</b><br>{rationale}</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.warning("⚠️ Could not parse JSON output — showing raw text below:")
                st.code(llm_output, language="json")
                st.session_state.output_text = llm_output

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()


# -------------------------------------------------------------------------
# Download Section
# -------------------------------------------------------------------------
if st.session_state.output_text:
    st.markdown("---")
    st.subheader("💾 Download Your Results")

    file_type = st.radio(
        "Select format:",
        options=["Text (.txt)", "PDF (.pdf)"],
        horizontal=True,
        key="download_option"
    )

    output_text = st.session_state.output_text

    if file_type == "Text (.txt)":
        st.download_button(
            label="⬇️ Download as .txt",
            data=output_text,
            file_name="clinical_summary.txt",
            mime="text/plain",
        )
    else:
        clean_text = sanitize_text(output_text)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "Clinical Report Summary", align="C")
        pdf.ln(10)
        for line in clean_text.splitlines():
            pdf.multi_cell(0, 10, line)
        pdf_output = pdf.output(dest="S").encode("latin1")
        st.download_button(
            label="⬇️ Download as .pdf",
            data=pdf_output,
            file_name="clinical_summary.pdf",
            mime="application/pdf",
        )

    st.caption(f"🧩 Embedding vector length: {st.session_state.embedding_dim}")


# -------------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------------
st.markdown(
    "<div class='footer'><b>© 2025 Udbhav Hackathon | Designed by Team Ashtrix & Co</b></div>",
    unsafe_allow_html=True,
)
