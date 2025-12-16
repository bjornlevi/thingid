import datetime as dt
import re
from typing import Optional


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%d.%m.%Y",
)


def parse_date(val: Optional[str]) -> Optional[dt.datetime]:
    """Parse a date/datetime string using common Alþingi formats."""
    if not val:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(val, fmt)
        except Exception:
            continue
    return None


def business_days_between(start: dt.date, end: dt.date) -> int:
    """
    Count business days between start and end (inclusive), minus 1 to match earlier behavior.
    Returns 0 when start >= end or when only weekends are in range.
    """
    if end < start:
        return 0
    days = 0
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days += 1
        cur += dt.timedelta(days=1)
    return max(days - 1, 0)


def prefer_athugasemd_date(text: Optional[str]) -> Optional[dt.datetime]:
    """
    Extract a date like 10.11.2025 from athugasemd text if present.
    """
    if not text:
        return None
    m = re.search(r"(\d{2})[.\-/](\d{2})[.\-/](\d{4})", text)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(0), "%d.%m.%Y")
    except Exception:
        return None
