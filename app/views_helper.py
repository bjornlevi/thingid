from __future__ import annotations

import datetime as dt
import json
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models

def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def parse_date(val: Optional[str]) -> Optional[dt.datetime]:
    if not val:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%d.%m.%Y"):
        try:
            return dt.datetime.strptime(val, fmt)
        except Exception:
            continue
    return None


# Icelandic alphabet order for sorting (lowercase)
_IS_ALPHA = "aábcdðeéfghiíjklmnoóprstuúvxyýþæö"
_IS_ORDER = {ch: idx for idx, ch in enumerate(_IS_ALPHA)}


def icelandic_sort_key(s: Optional[str]) -> List[int]:
    """
    Produce a sort key respecting Icelandic alphabet ordering.
    Falls back to ASCII order for unknown chars.
    """
    if not s:
        return []
    return [_IS_ORDER.get(ch.lower(), ord(ch)) for ch in s]


def current_lthing(session: Session) -> Optional[int]:
    try:
        val = session.execute(
            select(models.ThingmalalistiMal.attr_thingnumer).limit(1)
        ).scalar_one_or_none()
        return int(val) if val is not None else None
    except Exception:
        return None


def attach_flutningsmenn(doc: Any) -> None:
    """Parse and attach flutningsmenn JSON to _flutningsmenn attribute."""
    parsed_fm = []
    if getattr(doc, "leaf_kalladaftur", None):
        try:
            loaded = json.loads(doc.leaf_kalladaftur)
            if isinstance(loaded, list):
                parsed_fm = loaded
        except Exception:
            parsed_fm = []
    doc._flutningsmenn = parsed_fm


def flutningsmenn_primary_id(doc: Any) -> Optional[int]:
    """Return member id for flutningsmaður nr 1 (order==1) if present."""
    fallback = None
    for fm in getattr(doc, "_flutningsmenn", []):
        url = fm.get("profile_url") if isinstance(fm, dict) else None
        order = fm.get("order") if isinstance(fm, dict) else None
        if not url:
            continue
        try:
            qs = parse_qs(urlparse(url).query)
            if "nr" in qs:
                mid = int(qs["nr"][0])
                if order in (1, "1"):
                    return mid
                if fallback is None:
                    fallback = mid
        except Exception:
            continue
    return fallback
