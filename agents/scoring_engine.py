"""
Scoring Engine.
Scores each candidate against the JD using:
1. LLM reasoning (primary) — produces dimension scores with justifications
2. Embedding similarity (secondary) — cosine similarity for skills match signal

Pydantic validation enforces score bounds on every response.
"""

import json
import logging
import os
import numpy as np
from groq import Groq
from models.schemas import ParsedJD, ParsedProfile, CandidateScore, DimensionScore
from utils.security import sanitize_input

logger = logging.getLogger(__name__)

_client = None
_embedder = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _get_embedder():
    """Lazy-load SentenceTransformer — only loaded once."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer loaded: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not installed. Embedding signal disabled.")
            _embedder = False
    return _embedder if _embedder else None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _embedding_skills_signal(jd: ParsedJD, profile: ParsedProfile) -> float | None:
    """
    Returns a 0-10 embedding similarity signal for skills overlap.
    Returns None if embedder is unavailable.
    """
    embedder = _get_embedder()
    if not embedder:
        return None

    jd_text = " ".join(jd.required_skills + jd.preferred_skills)
    candidate_text = " ".join(profile.skills)

    if not jd_text.strip() or not candidate_text.strip():
        return None

    embs = embedder.encode([jd_text, candidate_text])
    sim = _cosine_similarity(embs[0], embs[1])
    return round(sim * 10, 1)


_SYSTEM_PROMPT = """You are a senior HR evaluator. Score a candidate against a job description 
using a strict rubric. Return ONLY valid JSON — no markdown, no explanation.

RUBRIC DIMENSIONS AND WEIGHTS:
- skills_match (30%): How well candidate's skills match JD required + preferred skills
  0=<30% match, 5=50-70% match, 10=>85% match
- experience_relevance (25%): Relevance of work history to this role
  0=Unrelated domain, 5=Adjacent domain, 10=Exact domain & seniority
- education_certs (15%): Education and certifications vs JD requirements
  0=Doesn't meet minimum, 5=Meets minimum, 10=Exceeds + extra certs
- project_portfolio (20%): Evidence of relevant projects/portfolio
  0=No evidence, 5=1-2 generic projects, 10=Strong relevant portfolio
- communication_quality (10%): Quality of writing, structure, clarity in resume
  0=Poor structure/grammar, 5=Adequate clarity, 10=Crisp, structured, impactful

Return this JSON schema:
{
  "skills_match": {"score": float 0-10, "justification": "one concise sentence"},
  "experience_relevance": {"score": float 0-10, "justification": "one concise sentence"},
  "education_certs": {"score": float 0-10, "justification": "one concise sentence"},
  "project_portfolio": {"score": float 0-10, "justification": "one concise sentence"},
  "communication_quality": {"score": float 0-10, "justification": "one concise sentence"}
}

Be objective. Base scores ONLY on evidence in the candidate profile. Do NOT fabricate skills."""


def score_candidate(jd: ParsedJD, profile: ParsedProfile) -> CandidateScore:
    """
    Score a single candidate against the JD.
    Combines LLM reasoning with optional embedding signal.
    """
    # Build the context for the LLM
    jd_summary = (
        f"Job Title: {jd.job_title}\n"
        f"Domain: {jd.domain}\n"
        f"Seniority: {jd.seniority_level}\n"
        f"Min Experience: {jd.min_experience_years} years\n"
        f"Required Education: {jd.required_education}\n"
        f"Required Skills: {', '.join(jd.required_skills)}\n"
        f"Preferred Skills: {', '.join(jd.preferred_skills)}\n"
        f"Key Responsibilities: {'; '.join(jd.key_responsibilities[:5])}"
    )

    profile_summary = (
        f"Candidate: {profile.candidate_name}\n"
        f"Experience: {profile.total_experience_years} years\n"
        f"Skills: {', '.join(profile.skills)}\n"
        f"Education: {'; '.join(profile.education)}\n"
        f"Certifications: {'; '.join(profile.certifications) or 'None'}\n"
        f"Work History: {'; '.join(profile.work_history[:5])}\n"
        f"Projects: {'; '.join(profile.projects[:5]) or 'None'}\n"
        f"Summary: {profile.summary}"
    )

    # Sanitize before sending to LLM
    jd_summary = sanitize_input(jd_summary)
    profile_summary = sanitize_input(profile_summary)

    # Get embedding signal (optional augmentation)
    emb_signal = _embedding_skills_signal(jd, profile)

    embedding_note = ""
    if emb_signal is not None:
        embedding_note = (
            f"\n\nEmbedding similarity signal for skills: {emb_signal}/10. "
            f"Use this as a soft reference — your reasoning takes precedence."
        )

    client = _get_client()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"JOB DESCRIPTION:\n{jd_summary}\n\n"
                    f"CANDIDATE PROFILE:\n{profile_summary}"
                    f"{embedding_note}"
                )
            }
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
        logger.error(f"Scoring JSON error for {profile.candidate_name}: {e}")
        raise ValueError(f"Scoring LLM returned invalid JSON: {e}")

    try:
        return CandidateScore(
            skills_match=DimensionScore(**data["skills_match"]),
            experience_relevance=DimensionScore(**data["experience_relevance"]),
            education_certs=DimensionScore(**data["education_certs"]),
            project_portfolio=DimensionScore(**data["project_portfolio"]),
            communication_quality=DimensionScore(**data["communication_quality"]),
        )
    except Exception as e:
        logger.error(f"Score validation error for {profile.candidate_name}: {e}")
        raise ValueError(f"Score validation failed: {e}")
