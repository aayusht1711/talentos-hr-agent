# 🎯 TalentOS — HR Resume & LinkedIn Shortlisting Agent

An AI agent that helps HR teams evaluate candidates at scale — objectively, consistently, and with full transparency. Built for the AI Enablement Internship — Task 1.

---

## Demo

```bash
streamlit run app.py
```

Upload a Job Description + resumes → ranked shortlist with rubric scores, radar chart, skills gap, and tailored interview questions in under 60 seconds.

---

## Problem

HR teams screening hundreds of applications per role face:
- **Fatigue** → inconsistent evaluation quality over time
- **Bias** → unconscious preference for familiar backgrounds  
- **Speed** → bottleneck on hiring timelines

This agent standardises evaluation against a fixed 5-dimension rubric, explains every score with a one-line justification, and keeps the human in the loop for final decisions.

---

## Agent Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│ Job Description │    │ Resume Batch     │    │ LinkedIn JSON        │
│ PDF / TXT       │    │ PDF / DOCX / TXT │    │ Manual export        │
└────────┬────────┘    └────────┬─────────┘    └──────────┬───────────┘
         │                      └────────────┬─────────────┘
         ▼                                   ▼
  ┌─────────────┐                 ┌─────────────────────┐
  │  JD Parser  │                 │  Profile Ingestion  │
  │  (LLM)      │                 │  Agent (LLM)        │
  └──────┬──────┘                 └──────────┬──────────┘
         └──────────────┬───────────────────┘
                        ▼
           ┌────────────────────────────┐
           │      Scoring Engine        │
           │  LLM reasoning (primary)   │
           │  Embedding similarity (aux)│
           │  Pydantic output validation│
           └────────────┬───────────────┘
                        ▼
              ┌──────────────────┐
              │  Ranker          │
              │  Sort + Shortlist│
              └────────┬─────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
  ┌──────────────────┐    ┌──────────────────────┐
  │  Report Generator│    │  HR Override Hook    │
  │  HTML / JSON /CSV│    │  Audit Log           │
  └──────────────────┘    └──────────────────────┘
```

**Agent pattern:** Sequential chain — 5 discrete steps, each validated before the next runs.  
No ReAct loop needed: the task is deterministic (parse → score → rank), not exploratory.

---

## Scoring Rubric

| Dimension | Weight | 0 — Poor | 5 — Average | 10 — Excellent |
|---|---|---|---|---|
| Skills Match | 30% | <30% overlap | 50–70% overlap | >85% overlap |
| Experience Relevance | 25% | Unrelated domain | Adjacent domain | Exact domain & seniority |
| Education & Certs | 15% | Below minimum | Meets minimum | Exceeds + extra certs |
| Project / Portfolio | 20% | No evidence | 1–2 generic projects | Strong relevant portfolio |
| Communication Quality | 10% | Poor structure | Adequate clarity | Crisp, structured, impactful |

**Hire thresholds:** ≥ 8.0 → Strong Hire · ≥ 6.5 → Hire · ≥ 5.0 → Maybe · < 5.0 → No Hire

---

## Technical Stack & Decision Log

### LLM: Llama 3.3 70B via Groq (`llama-3.3-70b-versatile`)

**Why Llama 3.3 70B on Groq over alternatives?**

| Criterion | Llama 3.3 70B (Groq) | GPT-4o | Gemini 1.5 Pro |
|---|---|---|---|
| Structured JSON output | ✅ Excellent | ✅ Good | ✅ Good |
| Inference speed | ✅ ~300 tok/s (Groq) | ❌ Slower | ⚠ Variable |
| Cost | ✅ Free tier generous | ❌ Expensive | ⚠ Moderate |
| Context window | 128K tokens | 128K tokens | 1M tokens |
| Privacy (local inference option) | ✅ Can run locally | ❌ Cloud only | ❌ Cloud only |
| Open source | ✅ Yes | ❌ No | ❌ No |

Groq's LPU hardware gives near-instant responses — critical for scoring 10+ resumes in a demo setting. The free tier is sufficient for this project with no cost risk.

### Agent Framework: Direct Groq SDK — Sequential Chain Pattern

**Architecture:** 5-step sequential pipeline (not ReAct loop, not multi-agent).

Each step is a discrete, validated agent function:
1. `jd_parser.parse_jd()` → structured `ParsedJD` (Pydantic)
2. `profile_parser.parse_profile()` → structured `ParsedProfile` per candidate
3. `scoring_engine.score_candidate()` → structured `CandidateScore` (LLM + embeddings)
4. `ranker.rank_candidates()` → sorted list of `CandidateResult`
5. `report_gen.generate_html_report()` → HTML / JSON / CSV output

**Why not LangChain/CrewAI?** For a linear, deterministic pipeline, direct SDK calls with Pydantic validation give tighter control over output format and error handling. No hidden abstractions that could mask scoring errors. Simpler to debug, audit, and explain to reviewers.

### Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

- Runs **locally** — zero extra API calls, zero cost
- Used as a *soft signal* for skills overlap (cosine similarity 0–10)
- LLM reasoning always takes precedence; embedding is advisory only
- Graceful fallback to LLM-only if sentence-transformers not installed

### Resume Parsing: `pdfplumber` + `python-docx`

- `pdfplumber`: Layout-aware text extraction for complex PDF layouts
- `python-docx`: Native DOCX paragraph extraction
- LinkedIn: Flat JSON → readable text converter built-in (`utils/file_loader.py`)

### Output Formats

- **HTML**: Fully self-contained report (inline CSS) — opens in any browser, shareable
- **JSON**: Structured export for ATS/HR system integration
- **CSV**: Excel-compatible with all dimension scores and skills gap per candidate

### UI: Streamlit

Clean professional SaaS-style interface with Inter font, radar chart comparison (Plotly), collapsible candidate cards, HR override controls, and LangSmith tracing integration.

---

## Prompt Design

### Iteration history

**JD Parser v1 (discarded):** Free-form text response — too inconsistent for downstream parsing.  
**v2:** Asked for JSON but allowed markdown fences — required extra stripping logic.  
**v3 (current):** Explicit "Return ONLY a JSON object — no markdown, no explanation" with schema inline. Added "Extract ONLY information present in the JD" guardrail after testing found the model inventing preferred skills.

**Scoring Engine v1 (discarded):** Single prompt asking for overall score — no dimension breakdown, no justifications.  
**v2:** Added dimension rubric anchors (0/5/10 descriptions) — scores became consistent.  
**v3 (current):** Added embedding similarity as a soft reference signal passed in the prompt, with explicit "your reasoning takes precedence" instruction to prevent over-reliance on the embedding score.

### JD Parser system prompt (key guardrails)

```
"Extract ONLY information present in the JD. Do NOT invent or infer."
"required_skills: must-have technical/soft skills explicitly stated"
"Return valid JSON only — no markdown, no explanation."
```

### Profile Parser system prompt (key guardrails)

```
"candidate_name: If not found, use 'Unknown Candidate'. Never fabricate."
"total_experience_years: Sum of all work experience. Use 0 if none found."
"Return valid JSON only."
```

### Scoring Engine system prompt (key guardrails)

```
"Base scores ONLY on evidence in the candidate profile. Do NOT fabricate skills."
"Embedding similarity signal ... Use this as a soft reference — your reasoning takes precedence."
```

---


## Prompt Iteration Log

Mentors want to see the thought process, not just the final prompt.

**JD Parser:**
- v1: Free-form text response → too inconsistent, downstream parsing broke
- v2: Asked for JSON but allowed markdown fences → required extra stripping
- v3 (current): Added `"Return ONLY a JSON object — no markdown"` + `"Do NOT invent skills"` guardrail after testing found the model fabricating preferred skills

**Scoring Engine:**
- v1: Single prompt for overall score → no dimension breakdown, not useful
- v2: Added 5 dimensions with rubric anchors (0/5/10 descriptions) → consistent scores
- v3 (current): Added embedding cosine similarity as a soft reference in the prompt with explicit `"your reasoning takes precedence"` instruction to prevent over-reliance on the embedding

## Security Mitigations

| Risk | Code Location | Mitigation Implemented |
|---|---|---|
| **Prompt Injection** | `utils/security.py` | 10 regex patterns detect injection signatures; matched content replaced with `[REDACTED]` before any LLM call |
| **Data Privacy / PII** | `utils/security.py` | `mask_pii_for_log()` strips emails and phone numbers before writing to `hr_agent.log`; full PII stays in memory only |
| **API Key Exposure** | `.env` + `.gitignore` | `.env` in `.gitignore`; `python-dotenv` loads keys; UI accepts key via password field; no keys hardcoded anywhere in codebase |
| **Hallucination Risk** | `models/schemas.py` | All LLM outputs validated by Pydantic v2 with strict field types and score bounds (`ge=0, le=10`); `ValueError` raised and surfaced to UI on failure |
| **Bias in Evaluation** | `app.py` | JD scanned for 14 known biased terms before evaluation; warning displayed with flagged terms |
| **Unauthorised Access** | `app.py` (Groq key check) | API key required at session start; for production: wrap behind FastAPI + OAuth2 + rate limiting |
| **Email Spoofing** | N/A — Task 1 | Not applicable; no email sending in Task 1 |

---

## Project Structure

```
hr-agent/
├── app.py                      # Streamlit UI — main entry point
├── agents/
│   ├── jd_parser.py            # Step 1: JD extraction agent
│   ├── profile_parser.py       # Step 2: Resume/LinkedIn profile parser
│   ├── scoring_engine.py       # Step 3: LLM + embedding scoring
│   ├── ranker.py               # Step 4: Sort + override logic + audit log
│   ├── report_gen.py           # Step 5: HTML, JSON report generation
│   └── interview_gen.py        # Bonus: Tailored interview question generator
├── models/
│   └── schemas.py              # Pydantic v2 models — all structured outputs
├── utils/
│   ├── file_loader.py          # PDF/DOCX/JSON text extraction
│   └── security.py             # Sanitization, PII masking, file validation
├── sample_data/
│   ├── jd_sample.txt           # Sample Senior ML Engineer JD
│   └── resumes/                # 5 sample resumes (strong → weak match)
│       ├── 01_arjun_mehta_strong.txt
│       ├── 02_priya_sharma_good.txt
│       ├── 03_ravi_krishnan_partial.txt
│       ├── 04_sneha_patel_adjacent.txt
│       └── 05_amit_verma_weak.txt
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
# 1. Clone and enter directory
git clone <repo_url>
cd hr-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env — add your GROQ_API_KEY from console.groq.com (free)

# 5. Run
python -m streamlit run app.py
```

Then in the sidebar: enter your Groq API key → click "Load sample JD" → click "Load 5 sample resumes" → click "Run shortlisting pipeline".

---

## Sample Output

Tested against Senior ML Engineer JD with 5 diverse resumes:

| Rank | Candidate | Score | Recommendation |
|---|---|---|---|
| 1 | Arjun Mehta | 9.2/10 | 🟢 Strong Hire |
| 2 | Priya Sharma | 6.8/10 | 🔵 Hire |
| 3 | Ravi Krishnan | 5.1/10 | 🟡 Maybe |
| 4 | Sneha Patel | 4.6/10 | 🔴 No Hire |
| 5 | Amit Verma | 1.2/10 | 🔴 No Hire |

---

## Model & Framework Disclosures

- **LLM:** `llama-3.3-70b-versatile` (Meta, via Groq)
- **Agent pattern:** Sequential chain (parse → score → rank)
- **Embeddings:** `all-MiniLM-L6-v2` (local, via sentence-transformers)
- **Output validation:** Pydantic v2 strict models
- **Resume parsing:** pdfplumber + python-docx
- **Observability:** LangSmith tracing (optional, via sidebar key input)
- **UI:** Streamlit 1.38+

---


