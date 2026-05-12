"""
JD Parser Agent.
Sends the raw job description text to Groq and parses structured requirements.
Uses Pydantic for strict output validation.
"""

import json
import logging
import os
from groq import Groq
from models.schemas import ParsedJD
from utils.security import sanitize_input

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


_SYSTEM_PROMPT = """You are a precise HR analyst. Your ONLY job is to extract structured 
information from a job description and return it as valid JSON.

Return ONLY a JSON object with this exact schema — no markdown, no explanation:
{
  "job_title": string,
  "required_skills": [string, ...],
  "preferred_skills": [string, ...],
  "min_experience_years": integer,
  "required_education": string,
  "key_responsibilities": [string, ...],
  "domain": string,
  "seniority_level": string  // "Junior" | "Mid" | "Senior" | "Lead" | "Executive"
}

Rules:
- Extract ONLY information present in the JD. Do NOT invent or infer missing data.
- required_skills: must-have technical/soft skills explicitly stated
- preferred_skills: nice-to-have skills (look for "preferred", "bonus", "plus")
- domain: industry or technical domain (e.g. "FinTech", "Machine Learning", "DevOps")
- Return valid JSON only."""


def parse_jd(jd_text: str) -> ParsedJD:
    """
    Parse a job description into a structured ParsedJD object.
    Raises ValueError if LLM output cannot be validated.
    """
    clean_text = sanitize_input(jd_text)
    if not clean_text.strip():
        raise ValueError("Job description is empty after sanitization.")

    client = _get_client()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this job description:\n\n{clean_text}"}
        ]
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"JD parser JSON decode error: {e}\nRaw: {raw[:500]}")
        raise ValueError(f"LLM returned invalid JSON: {e}")

    try:
        return ParsedJD(**data)
    except Exception as e:
        logger.error(f"JD parser Pydantic validation error: {e}")
        raise ValueError(f"JD structured output validation failed: {e}")
