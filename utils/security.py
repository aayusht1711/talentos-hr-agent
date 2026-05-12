"""
Security utilities.
- Input sanitization to prevent prompt injection
- PII masking for logs (emails, phone numbers)
- Safe text extraction
"""

import re
import html


# Patterns that could signal prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"disregard\s+all\s+prior",
    r"you\s+are\s+now\s+a",
    r"system\s*:\s*you",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"<\|im_start\|>",
    r"forget\s+everything",
    r"new\s+persona",
    r"jailbreak",
]

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")


def sanitize_input(text: str) -> str:
    """
    Sanitize raw text before sending to LLM.
    - Escape HTML entities
    - Strip prompt injection attempts (replace with [REDACTED])
    - Limit length to prevent token flooding
    """
    if not text:
        return ""

    # Decode html entities then re-escape (normalise)
    text = html.unescape(text)

    # Strip null bytes and control characters (except newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Check for injection patterns
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)

    # Truncate to 12,000 chars per document (generous for resumes, prevents flooding)
    return text[:12000]


def mask_pii_for_log(text: str) -> str:
    """
    Mask emails and phone numbers before writing to log files.
    Keeps names intact (needed for HR context) but removes contact details.
    """
    text = _EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
    text = _PHONE_PATTERN.sub("[PHONE_REDACTED]", text)
    return text


def validate_file_extension(filename: str, allowed: list[str]) -> bool:
    """Check that an uploaded file has an allowed extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in [a.lower().strip(".") for a in allowed]
