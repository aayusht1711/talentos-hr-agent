"""
Profile Parser Agent.
Converts raw resume text (from PDF/DOCX/LinkedIn JSON) into a structured ParsedProfile.
"""

import json
import logging
import os
from groq import Groq
from models.schemas import ParsedProfile
from utils.security import sanitize_input

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


_SYSTEM_PROMPT = """You are a precise resume parser. Extract structured information from 
a candidate's resume or LinkedIn profile and return valid JSON only.

Return ONLY a JSON object with this exact schema:
{
  "candidate_name": string,
  "email": string or null,
  "skills": [string, ...],
  "total_experience_years": float,
  "education": [string, ...],
  "certifications": [string, ...],
  "work_history": [string, ...],
  "projects": [string, ...],
  "summary": string
}

Rules:
- candidate_name: Full name only. If not found, use "Unknown Candidate".
- email: extract if present, else null. Never fabricate.
- skills: ALL technical and soft skills mentioned anywhere in the document.
- total_experience_years: Sum of all work experience. Use 0 if none found.
- education: Each entry as "<Degree> in <Field> from <Institution> (<Year>)"
- work_history: Each entry as "<Title> at <Company> (<Duration>): <1 line summary>"
- projects: Each as "<Project Name>: <1 line description>"
- summary: 2-3 sentence professional summary of the candidate.
- Return valid JSON only — no markdown, no explanation."""


def parse_profile(raw_text: str, source_file: str = "") -> ParsedProfile:
    """
    Parse a resume or LinkedIn profile text into a structured ParsedProfile.
    """
    clean_text = sanitize_input(raw_text)
    if not clean_text.strip():
        raise ValueError(f"Profile text is empty for: {source_file}")

    client = _get_client()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this candidate profile:\n\n{clean_text}"}
        ]
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Profile parser JSON error for {source_file}: {e}")
        raise ValueError(f"LLM returned invalid JSON for {source_file}: {e}")

    try:
        profile = ParsedProfile(**data, source_file=source_file)
        return profile
    except Exception as e:
        logger.error(f"Profile validation error for {source_file}: {e}")
        raise ValueError(f"Profile validation failed for {source_file}: {e}")
