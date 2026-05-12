"""
Pydantic models for all structured outputs.
Using strict validation to prevent hallucinations from LLM responses.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum


class HireRecommendation(str, Enum):
    STRONG_HIRE = "Strong Hire"
    HIRE = "Hire"
    MAYBE = "Maybe"
    NO_HIRE = "No Hire"


class DimensionScore(BaseModel):
    score: float = Field(..., ge=0, le=10, description="Score from 0 to 10")
    justification: str = Field(..., min_length=10, max_length=300)

    @validator("score")
    def round_score(cls, v):
        return round(v, 1)


class CandidateScore(BaseModel):
    skills_match: DimensionScore
    experience_relevance: DimensionScore
    education_certs: DimensionScore
    project_portfolio: DimensionScore
    communication_quality: DimensionScore

    @property
    def weighted_total(self) -> float:
        total = (
            self.skills_match.score * 0.30
            + self.experience_relevance.score * 0.25
            + self.education_certs.score * 0.15
            + self.project_portfolio.score * 0.20
            + self.communication_quality.score * 0.10
        )
        return round(total, 2)

    @property
    def hire_recommendation(self) -> HireRecommendation:
        t = self.weighted_total
        if t >= 8.0:
            return HireRecommendation.STRONG_HIRE
        elif t >= 6.5:
            return HireRecommendation.HIRE
        elif t >= 5.0:
            return HireRecommendation.MAYBE
        else:
            return HireRecommendation.NO_HIRE


class ParsedJD(BaseModel):
    job_title: str
    required_skills: List[str] = Field(..., min_items=1)
    preferred_skills: List[str] = Field(default_factory=list)
    min_experience_years: int = Field(..., ge=0)
    required_education: str
    key_responsibilities: List[str] = Field(..., min_items=1)
    domain: str
    seniority_level: str


class ParsedProfile(BaseModel):
    candidate_name: str
    email: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    total_experience_years: float = Field(..., ge=0)
    education: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    work_history: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    summary: str = ""
    source_file: str = ""


class CandidateResult(BaseModel):
    profile: ParsedProfile
    scores: CandidateScore
    rank: int = 0
    override_applied: bool = False
    override_reason: Optional[str] = None

    @property
    def weighted_total(self) -> float:
        return self.scores.weighted_total

    @property
    def hire_recommendation(self) -> HireRecommendation:
        return self.scores.hire_recommendation
