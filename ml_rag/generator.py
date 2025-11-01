import os
import json
from openai import OpenAI

def generate_summary_and_diagnosis(report_text: str) -> dict:
    """
    Calls the LLM to generate a concise summary and differential diagnoses.
    Ensures JSON-only response and handles empty input.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in .env")

    if not report_text.strip():
        return {
            "raw_response": "⚠️ No text extracted from PDF. Please check the uploaded file.",
            "parsed_response": None,
        }

    client = OpenAI(api_key=key)

    system_prompt = """You are a clinical documentation assistant.
Respond ONLY in JSON, no greeting or explanation.
Format:
{
  "summary": "<concise factual summary>",
  "differentials": [
    {"rank": 1, "diagnosis": "...", "rationale": "..."},
    {"rank": 2, "diagnosis": "...", "rationale": "..."}
  ]
}"""

    user_prompt = f"""Analyze this patient report and return the summary
and top 5 possible differential diagnoses (with one-line rationales).

Patient report:
{report_text}
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",        # more reliable for JSON
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    raw_output = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw_output)
    except Exception:
        parsed = None

    return {"raw_response": raw_output, "parsed_response": parsed}
