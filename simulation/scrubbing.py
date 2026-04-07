"""Scrubber — redacts secrets and PII before any data is written to the database.

One Scrubber instance is created per session (not per turn). Pass all decrypted
auth_config values at construction time so they are compiled into regex patterns once.
"""

import re


class Scrubber:
    """Redacts secrets and PII from strings before DB persistence.

    Args:
        secrets: List of decrypted secret values from project.auth_config.
                 Pre-compiled into regex patterns at construction time.
                 Never logged or stored beyond pattern compilation.
    """

    # PII patterns — compiled once at class definition time
    _EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    _PHONE_RE = re.compile(
        r"(?:"
        r"\+?1?\s?[\(\-]?\d{3}[\)\-\s]?\s?\d{3}[\-\s]?\d{4}"  # US: (555) 867-5309
        r"|"
        r"\+\d{1,3}[\s\-]?\d{1,4}[\s\-]?\d{1,4}[\s\-]?\d{1,9}"  # E.164 international
        r")"
    )

    def __init__(self, secrets: list[str]) -> None:
        # Build per-instance secret patterns; sort by length descending so longer
        # secrets are replaced before shorter substrings of them.
        self._secret_patterns: list[re.Pattern] = [
            re.compile(re.escape(s))
            for s in sorted(secrets, key=len, reverse=True)
            if s  # skip empty strings
        ]

    def scrub(self, text: str) -> str:
        """Replace secrets, emails, and phone numbers with redaction placeholders.

        Returns the sanitised string. Never raises — returns original text on error.
        """
        try:
            for pattern in self._secret_patterns:
                text = pattern.sub("[REDACTED_SECRET]", text)
            text = self._EMAIL_RE.sub("[REDACTED_EMAIL]", text)
            text = self._PHONE_RE.sub("[REDACTED_PHONE]", text)
            return text
        except Exception:
            return text
