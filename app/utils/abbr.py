import unicodedata


def norm_abbr(s: str) -> str:
    """Normalize a member abbreviation (skammstöfun) for matching, keeping Icelandic letters (Þ, Á, Ð, ...)."""
    s = unicodedata.normalize("NFC", s or "")
    return "".join(ch for ch in s if ch.isalnum()).lower()
