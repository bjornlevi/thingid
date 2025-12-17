"""Central constants, labels and normalization helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# Written questions (used for answer status / svartími)
WRITTEN_QUESTION_LABEL = "fyrirspurn til skrifl. svars"

# Answer status labels
ANSWER_STATUS_SVARAD = "svarað"
ANSWER_STATUS_OSVARAD = "ósvarað"


def _norm_label(val: Optional[str]) -> str:
    if not val:
        return ""
    s = val.strip()
    s = s.replace("đ", "ð").replace("Đ", "Ð")
    s = s.replace("ţ", "þ").replace("Ţ", "Þ")
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s


def canonical_issue_type(val: Optional[str]) -> str:
    """
    Canonicalize issue type labels for stable filtering/counting.
    Keeps original casing for known Icelandic labels, otherwise returns normalized value.
    """
    s = _norm_label(val)
    k = s.casefold()

    mapping = {
        "frumvarp til laga": "Frumvarp til laga",
        "lagafrumvarp": "Frumvarp til laga",
        "tillaga til þingsályktunar": "Tillaga til þingsályktunar",
        "þingsályktunartillaga": "Tillaga til þingsályktunar",
        "beiðni um skýrslu": "Beiðni um skýrslu",
        "beiđni um skýrslu": "Beiðni um skýrslu",
        "skýrsla": "Skýrsla",
        "fyrirspurn": "Fyrirspurn",
        "óundirbúinn fyrirspurnatími": "óundirbúinn fyrirspurnatími",
        "sérstök umræða": "sérstök umræða",
        "álit": "Álit",
        WRITTEN_QUESTION_LABEL.casefold(): WRITTEN_QUESTION_LABEL,
    }
    return mapping.get(k, s)
