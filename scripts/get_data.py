#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
get_data.py
Uses generated models.py + schema_map.json (from check_data.py) to load Alþingi XML into SQLite.

Usage:
  pip install requests sqlalchemy
  python get_data.py --db althingi.db
  python get_data.py --db althingi.db --schema schema_map.json
  python get_data.py --db althingi.db --max-records 200
  python get_data.py --models-dir app
  python get_data.py --force-fetch  # bypass cache
  python get_data.py --no-reset-db  # keep existing DB file
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import sys
import time
import re
import unicodedata
from html import unescape
from html.parser import HTMLParser
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qs
from pathlib import Path

# Ensure project root is on sys.path for local imports (app.*)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
import xml.etree.ElementTree as ET
from sqlalchemy import UniqueConstraint, create_engine, text, select, event
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from app.constants import WRITTEN_QUESTION_LABEL, ANSWER_STATUS_SVARAD, ANSWER_STATUS_OSVARAD
from app.utils.dates import parse_date, business_days_between, prefer_athugasemd_date

def _norm_tag(tag: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "", tag or "")
    return cleaned.lower()

BASE_CURRENT = "https://www.althingi.is/altext/xml/loggjafarthing/yfirstandandi/"
BASE_ALL = "https://www.althingi.is/altext/xml/loggjafarthing/"


# -----------------------------
# Helpers
# -----------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _filter_model_kwargs(Model: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    """
    SQLAlchemy model constructors reject unknown keyword arguments.
    Be defensive and drop unexpected keys (e.g. raw XML tag names) to avoid hard crashes.
    """
    try:
        valid = {c.key for c in Model.__mapper__.column_attrs}
    except Exception:
        valid = set()
    if not valid:
        return row
    return {k: v for k, v in row.items() if k in valid}

class Fetcher:
    def __init__(self, timeout: int = 30, sleep_s: float = 0.15, cache_dir: Optional[str] = None, max_age_hours: float = 23, force: bool = False):
        self.sess = requests.Session()
        self.sess.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; althingi-ingest/1.0)"
        })
        self.timeout = timeout
        self.sleep_s = sleep_s
        self.cache_dir = cache_dir
        self.max_age_hours = max_age_hours
        self.force = force
        self.cache_only_default = False
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, url: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", url)
        return os.path.join(self.cache_dir, safe)

    def get(self, url: str, retries: int = 4, cache_only: Optional[bool] = None) -> bytes:
        cache_path = self._cache_path(url)
        if cache_only is None:
            cache_only = self.cache_only_default
        if cache_path and not self.force and os.path.exists(cache_path):
            age_hours = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))).total_seconds() / 3600.0
            if cache_only or age_hours < self.max_age_hours:
                with open(cache_path, "rb") as f:
                    return f.read()

        last = None
        for i in range(retries):
            try:
                r = self.sess.get(url, timeout=self.timeout)
                r.raise_for_status()
                time.sleep(self.sleep_s)
                content = r.content
                if cache_path:
                    with open(cache_path, "wb") as f:
                        f.write(content)
                return content
            except Exception as e:
                last = e
                time.sleep(0.5 * (i + 1))
        raise RuntimeError(f"Failed to fetch {url}: {last}") from last

def parse_xml(content: bytes, url: str) -> ET.Element:
    try:
        return ET.fromstring(content)
    except Exception as e:
        raise RuntimeError(f"XML parse failed for {url}: {e}") from e

def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag

def is_abs_url(u: str) -> bool:
    p = urlparse(u)
    return bool(p.scheme and p.netloc)

def norm_url(base: str, u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    return u if is_abs_url(u) else urljoin(base, u)


def parse_flutningsmenn(detail_xml: ET.Element) -> List[Dict[str, Any]]:
    """
    Extract ordered flutningsmenn entries from a thingskjal detail XML.
    Returns list of dicts with order, name, role (ráðherra), and profile_url if present.
    """
    out: List[Dict[str, Any]] = []
    for fm in detail_xml.iter():
        if strip_ns(fm.tag) != "flutningsmaður":
            continue
        order = None
        try:
            order = int(fm.attrib.get("röð") or fm.attrib.get("rod") or fm.attrib.get("rod") or fm.attrib.get("roed"))
        except Exception:
            pass
        name = ""
        role = ""
        profile_url = ""
        for c in list(fm):
            tag = strip_ns(c.tag)
            if tag == "nafn":
                name = (c.text or "").strip()
            elif tag == "ráðherra":
                role = (c.text or "").strip()
            elif tag == "xml":
                profile_url = (c.text or "").strip()
        display_name = name or role
        if display_name:
            out.append({
                "order": order or 0,
                "name": display_name,
                "role": role,
                "profile_url": profile_url,
            })
    out.sort(key=lambda x: x["order"])
    return out

def iter_leaf_fields(record: ET.Element) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    def rec(node: ET.Element, prefix: str):
        children = [c for c in list(node) if isinstance(c.tag, str)]
        tag = strip_ns(node.tag)
        here = f"{prefix}/{tag}" if prefix else tag
        if not children:
            v = (node.text or "").strip()
            if v:
                out.append((here, v))
            return
        for c in children:
            rec(c, here)

    for c in list(record):
        rec(c, "")
    return out

def iter_records(root: ET.Element, record_path: str) -> List[ET.Element]:
    parts = record_path.split("/")
    if len(parts) == 1:
        tag = parts[0]
        return [e for e in list(root) if strip_ns(e.tag) == tag]
    if len(parts) == 2:
        cont_tag, item_tag = parts
        cont = None
        for c in list(root):
            if strip_ns(c.tag) == cont_tag:
                cont = c
                break
        if cont is None:
            return []
        return [e for e in list(cont) if strip_ns(e.tag) == item_tag]
    return []

def _parse_sessions(root: ET.Element) -> List[ET.Element]:
    sessions = []
    for e in root.iter():
        if strip_ns(e.tag) == "þing" and e.attrib:
            sessions.append(e)
    return sessions


def _extract_yfirlit(node: ET.Element, base_url: str) -> Dict[str, str]:
    yf = None
    for c in list(node):
        if strip_ns(c.tag) == "yfirlit":
            yf = c
            break
    if yf is None:
        return {}
    urls = {}
    for c in list(yf):
        name = strip_ns(c.tag)
        u = (c.text or "").strip()
        if u:
            urls[name] = norm_url(base_url, u)
    return urls


def discover_current_lthing_and_yfirlit(fetcher: Fetcher) -> Tuple[int, Dict[str, str]]:
    # Minimal parse: choose max <þing númer="..."> like in check_data
    root = parse_xml(fetcher.get(BASE_CURRENT), BASE_CURRENT)

    sessions = _parse_sessions(root)
    if not sessions:
        raise RuntimeError("No <þing> found in yfirstandandi")

    best_n = None
    best = None
    for s in sessions:
        n = s.attrib.get("númer") or s.attrib.get("numer") or s.attrib.get("nr")
        try:
            ni = int(str(n))
        except Exception:
            continue
        if best_n is None or ni > best_n:
            best_n, best = ni, s
    if best is None:
        raise RuntimeError("Could not determine current session number")

    urls = _extract_yfirlit(best, BASE_CURRENT)
    if not urls:
        raise RuntimeError("No <yfirlit> under chosen <þing>")

    return best_n, urls


def discover_lthing_and_yfirlit(fetcher: Fetcher, lthing: int) -> Tuple[int, Dict[str, str]]:
    root = parse_xml(fetcher.get(BASE_ALL), BASE_ALL)
    sessions = _parse_sessions(root)
    if not sessions:
        raise RuntimeError("No <þing> found in loggjafarthing")
    for s in sessions:
        n = s.attrib.get("númer") or s.attrib.get("numer") or s.attrib.get("nr")
        try:
            ni = int(str(n))
        except Exception:
            continue
        if ni == lthing:
            urls = _extract_yfirlit(s, BASE_ALL)
            return lthing, urls
    raise RuntimeError(f"Could not find lthing {lthing} in loggjafarthing")


def list_lthing_sessions(fetcher: Fetcher) -> List[Tuple[int, Dict[str, str]]]:
    root = parse_xml(fetcher.get(BASE_ALL), BASE_ALL)
    sessions = _parse_sessions(root)
    out = []
    for s in sessions:
        n = s.attrib.get("númer") or s.attrib.get("numer") or s.attrib.get("nr")
        try:
            ni = int(str(n))
        except Exception:
            continue
        urls = _extract_yfirlit(s, BASE_ALL)
        out.append((ni, urls))
    out.sort(key=lambda x: x[0])
    return out


def unique_constraint_columns(model: Any) -> List[List[str]]:
    """
    Return lists of column names participating in UniqueConstraints on the model.
    Used to drop duplicate records within the same ingestion batch before hitting the DB.
    """
    cols: List[List[str]] = []
    table = getattr(model, "__table__", None)
    if table is None:
        return cols
    for c in table.constraints:
        if isinstance(c, UniqueConstraint):
            cols.append([col.name for col in c.columns])
    return cols


def execute_with_retry(session: Session, stmt: Any, params: Optional[Dict[str, Any]] = None, retries: int = 10, delay: float = 2.0) -> None:
    """Execute a SQL statement with simple retry on SQLite 'database is locked' errors."""
    for attempt in range(retries):
        try:
            session.execute(stmt, params or {})
            return
        except OperationalError as e:
            msg = str(e).lower()
            if "database is locked" in msg or "database is busy" in msg:
                if attempt + 1 == retries:
                    raise
                time.sleep(delay)
                delay *= 1.2
                continue
            raise


def commit_with_retry(session: Session, retries: int = 10, delay: float = 2.0) -> None:
    """Commit with retry on SQLite busy/locked errors."""
    for attempt in range(retries):
        try:
            session.commit()
            return
        except OperationalError as e:
            msg = str(e).lower()
            if "database is locked" in msg or "database is busy" in msg:
                session.rollback()
                if attempt + 1 == retries:
                    raise
                time.sleep(delay)
                delay *= 1.2
                continue
            raise


def cache_nefndarfundir(fetcher: Fetcher, lthing: int) -> None:
    """
    Fetch nefndarfundir list and cache each fundargerð XML/HTML using the existing fetcher cache.
    """
    base_url = f"https://www.althingi.is//altext/xml/nefndarfundir/?lthing={lthing}"
    try:
        root = parse_xml(fetcher.get(base_url), base_url)
    except Exception as e:
        print(f"[warn] failed to fetch nefndarfundir for lthing {lthing}: {e}")
        return

    seen = set()
    fetched = 0
    for nf in root.iter():
        if strip_ns(nf.tag) != "nefndarfundur":
            continue
        for fg in nf.iter():
            if strip_ns(fg.tag) == "xml":
                u = (fg.text or "").strip()
                if not u:
                    continue
                full = norm_url(base_url, u)
                if full in seen:
                    continue
                seen.add(full)
                try:
                    fetcher.get(full)
                    fetched += 1
                except Exception as e:
                    print(f"[warn] failed to fetch fundargerð {full}: {e}")
    print(f"[ok] nefndarfundir: cached {fetched} fundargerðir for lthing {lthing}")


def parse_attendance_text(text: str, abbr_to_id: Dict[str, int]) -> Tuple[List[Dict[str, Any]], set]:
    """
    Parse attendance from fundargerð HTML-ish text.
    Returns (attendance_records, meeting_member_ids_seen)
    attendance_records: dicts with member_id, status, substitute_for_member_id (optional)
    """
    attendance: List[Dict[str, Any]] = []
    seen_members: set = set()
    if not text:
        return attendance, seen_members
    lower = text.lower()
    start = lower.find("mætt:")
    if start == -1:
        return attendance, seen_members
    section = text[start:]
    stop_markers = ["<h2", "bókað:", "b\u00f3ka\u00f0:"]
    stop = len(section)
    for m in stop_markers:
        idx = section.lower().find(m, 5)
        if idx != -1:
            stop = min(stop, idx)
    section = section[:stop]
    parts = re.split(r"<br\s*/?>", section, flags=re.IGNORECASE)
    for raw in parts:
        line = raw.strip()
        if not line:
            continue
        lower_line = line.lower()
        # notified absence (singular/plural)
        if "boðaði forföll" in lower_line or "bo\u00f0a\u00f0i forf\u00f6ll" in lower_line or "bo\u00f0u\u00f0u forf\u00f6ll" in lower_line or "bo\u00f0u\u00f0u forfoll" in lower_line or "bo\u00f0u\u00f0u forfall" in lower_line:
            abbrs = re.findall(r"\(([^\)]+)\)", line)
            if not abbrs:
                continue
            for ab in abbrs:
                abbr = _norm_tag(ab)
                mid = abbr_to_id.get(abbr)
                if mid and mid not in seen_members:
                    seen_members.add(mid)
                    attendance.append({"member_id": mid, "status": "absent_notified"})
            continue
        # proxy: ... fyrir Y (ABBR) ... only count target as present
        if "fyrir" in lower_line or "sat fundinn fyrir" in lower_line:
            abbrs = re.findall(r"\(([^\)]+)\)", line)
            target_mid = None
            proxy_mid = None
            if abbrs:
                if len(abbrs) >= 2:
                    proxy_mid = abbr_to_id.get(_norm_tag(abbrs[0]))
                    target_mid = abbr_to_id.get(_norm_tag(abbrs[-1]))
                else:
                    target_mid = abbr_to_id.get(_norm_tag(abbrs[-1]))
            if target_mid:
                if target_mid not in seen_members:
                    seen_members.add(target_mid)
                    attendance.append({"member_id": target_mid, "status": "present", "substitute_for_member_id": proxy_mid})
            continue
        # normal attendance
        m = re.search(r"\(([^\)]+)\)", line)
        if m:
            abbr = _norm_tag(m.group(1))
            mid = abbr_to_id.get(abbr)
            if mid:
                seen_members.add(mid)
            attendance.append({"member_id": mid, "status": "present"})
    return attendance, seen_members


def parse_attendance_from_html(text: str, abbr_to_id: Dict[str, int]) -> Tuple[List[Dict[str, Any]], set]:
    """
    Wrapper to parse attendance from fundargerð text (often HTML-ish).
    """
    return parse_attendance_text(text, abbr_to_id)


def parse_arrival_times(text: str, abbr_to_id: Dict[str, int]) -> Dict[int, str]:
    """
    Best-effort arrival time extraction per member_id from fundargerð text.
    Returns {member_id: "HH:MM"}
    """
    arrivals: Dict[int, str] = {}
    if not text:
        return arrivals
    lines = re.split(r"<br\s*/?>", text, flags=re.IGNORECASE)
    for line in lines:
        lower = unicodedata.normalize("NFKD", line.lower())
        lower = "".join(ch for ch in lower if not unicodedata.combining(ch))
        m_time = re.search(r"kl\.\s*(\d{2}:\d{2})", lower)
        if not m_time:
            m_time = re.search(r"(\d{2}:\d{2})", lower)
        if not m_time:
            continue
        time_str = m_time.group(1)
        abbrs = re.findall(r"\(([^\)]+)\)", lower)
        if not abbrs:
            continue
        for ab in abbrs:
            norm_ab = _norm_tag(ab)
            mid = abbr_to_id.get(norm_ab)
            if mid:
                arrivals[mid] = time_str
    return arrivals


def populate_vote_sessions_and_attendance(session: Session, lthing: int, manual_models: Any) -> Dict[int, Dict[str, int]]:
    """
    Store vote sessions (vote_num, time) and return per-member vote counts excluding notified absences.
    """
    counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {"attended": 0, "total": 0})
    if not manual_models or not hasattr(manual_models, "VoteSession"):
        return counts
    VoteSession = manual_models.VoteSession
    execute_with_retry(session, text('DELETE FROM vote_session WHERE lthing=:lt'), {"lt": lthing})
    session.flush()
    rows = session.execute(text("SELECT attr_atkvaedagreidslunumer, leaf_timi FROM atkvaedagreidslur__atkvaedagreidsla")).fetchall()
    seen_vote_nums = set()
    for vote_num, leaf_timi in rows:
        if vote_num is None or vote_num in seen_vote_nums:
            continue
        seen_vote_nums.add(vote_num)
        session.add(VoteSession(lthing=lthing, vote_num=vote_num, time=leaf_timi))
    commit_with_retry(session)

    # Build voter attendance counts from vote_details already stored
    rows = session.execute(
        text("SELECT voter_id, vote FROM vote_details WHERE vote IN ('já','nei','greiðir ekki atkvæði','boðaði fjarvist')")
    ).fetchall()
    for voter_id, vote in rows:
        if voter_id is None:
            continue
        if vote == "boðaði fjarvist":
            # counts as an opportunity but not attended
            counts[int(voter_id)]["total"] += 1
            continue
        counts[int(voter_id)]["attended"] += 1
        counts[int(voter_id)]["total"] += 1
    return counts


def populate_committee_attendance(session: Session, cache_dir: Path, lthing: int, abbr_to_id: Dict[str, int], manual_models: Any) -> Dict[int, Dict[str, int]]:
    """
    Parse cached fundargerðir and store committee_meeting + committee_attendance.
    Returns per-member counts {member_id: {"attended": x, "total": y}}
    """
    counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {"attended": 0, "total": 0})
    if not manual_models or not hasattr(manual_models, "CommitteeMeeting") or not hasattr(manual_models, "CommitteeAttendance"):
        return counts
    CommitteeMeeting = manual_models.CommitteeMeeting
    CommitteeAttendance = manual_models.CommitteeAttendance
    session.execute(text('DELETE FROM committee_attendance WHERE lthing=:lt'), {"lt": lthing})
    session.execute(text('DELETE FROM committee_meeting WHERE lthing=:lt'), {"lt": lthing})
    session.flush()

    matched = 0
    meetings_added = 0
    attendance_added = 0
    skipped_no_texti = 0
    skipped_no_attendance = 0
    parse_errors: List[Tuple[str, str]] = []

    for fp in cache_dir.glob("*nefndarfundur*"):
        matched += 1
        try:
            root = ET.parse(fp).getroot()
            attrs = root.attrib
            meeting_num = None
            try:
                meeting_num = int(attrs.get("númer") or attrs.get("numer") or attrs.get("nummer") or attrs.get("númer") or attrs.get("númer"))
            except Exception:
                pass
            nefnd_id = None
            for c in root:
                if strip_ns(c.tag) == "nefnd":
                    try:
                        nefnd_id = int(c.attrib.get("id"))
                    except Exception:
                        nefnd_id = None
                    break
            meeting_dt = None
            meeting_end = None
            for t in root.iter():
                tag = strip_ns(t.tag)
                if tag == "dagurtími" or tag == "dagurtimi":
                    meeting_dt = parse_date(t.text)
                elif tag == "fundursettur":
                    meeting_dt = meeting_dt or parse_date(t.text)
                elif tag == "fuslit":
                    meeting_end = parse_date(t.text)
            texti = ""
            for t in root.iter():
                if strip_ns(t.tag) == "texti":
                    candidate = t.text or ""
                    if len(candidate) > len(texti):
                        texti = candidate
            # store meeting only if fundargerð text is present; otherwise skip counting
            if not texti:
                skipped_no_texti += 1
                continue
            attendance, seen_ids = parse_attendance_from_html(texti, abbr_to_id)
            if not attendance:
                # no fundargerð attendance -> skip counting this meeting
                skipped_no_attendance += 1
                continue
            arrival_map = parse_arrival_times(texti, abbr_to_id)
            mt = CommitteeMeeting(
                meeting_num=meeting_num,
                lthing=lthing,
                nefnd_id=nefnd_id,
                start_time=meeting_dt.isoformat() if meeting_dt else None,
                end_time=meeting_end.isoformat() if meeting_end else None,
                raw_xml=ET.tostring(root, encoding="unicode"),
            )
            session.add(mt)
            session.flush()
            meeting_id = mt.id
            meetings_added += 1
            seen_attendance: set[Tuple[int, int]] = set()
            for rec in attendance:
                mid = rec.get("member_id")
                if not mid:
                    continue
                key = (meeting_id, int(mid))
                if key in seen_attendance:
                    continue
                seen_attendance.add(key)
                status = rec.get("status") or "present"
                arrival_val = arrival_map.get(mid)
                substitute_for = rec.get("substitute_for_member_id")
                session.add(CommitteeAttendance(
                    meeting_id=meeting_id,
                    meeting_num=meeting_num,
                    lthing=lthing,
                    member_id=mid,
                    status=status,
                    substitute_for_member_id=substitute_for,
                    arrival_time=arrival_val,
                ))
                attendance_added += 1
                if status == "present" or status == "proxy_present":
                    counts[mid]["attended"] += 1
                    counts[mid]["total"] += 1
                elif status == "absent_notified":
                    counts[mid]["total"] += 1
            commit_with_retry(session)
        except Exception as e:
            session.rollback()
            if len(parse_errors) < 5:
                parse_errors.append((str(fp), str(e)))
            continue

    if parse_errors:
        for fp, msg in parse_errors:
            print(f"[warn] committee_attendance parse failed for {fp}: {msg}")
    print(
        "[ok] committee_attendance:"
        f" matched {matched} cached fundargerðir,"
        f" stored {meetings_added} meetings and {attendance_added} attendance rows"
        f" (skipped: no_texti={skipped_no_texti}, no_attendance={skipped_no_attendance})"
    )
    return counts


def compute_issue_metrics(
    lthing: int,
    issue_docs: List[Any],
    issues: List[Any],
    athugasemd_by_skjalsnr: Dict[int, str],
    manual_models: Any,
) -> List[Any]:
    """Compute answer status/latency for written questions."""
    if not manual_models or not hasattr(manual_models, "IssueMetrics"):
        return []
    metrics_model = manual_models.IssueMetrics

    docs_by_mal: Dict[int, List[Any]] = defaultdict(list)
    first_doc_date: Dict[int, Any] = {}
    for d in issue_docs:
        if d.malnr is None:
            continue
        docs_by_mal[int(d.malnr)].append(d)
        utb_dt = parse_date(d.utbyting)
        if utb_dt:
            existing = first_doc_date.get(int(d.malnr))
            if existing is None or utb_dt < existing:
                first_doc_date[int(d.malnr)] = utb_dt

    metrics: List[Any] = []
    seen_keys: set[tuple] = set()
    for issue in issues:
        if getattr(issue, "attr_malsflokkur", None) != "A":
            continue
        typ = (issue.leaf_malstegund_heiti2 or issue.leaf_malstegund_heiti or "").strip().casefold()
        if typ != WRITTEN_QUESTION_LABEL.casefold():
            continue
        key = int(issue.attr_malsnumer)
        dedupe_key = (lthing, key)
        if dedupe_key in seen_keys:
            continue
        question_dt = None
        answer_dt = None
        for doc in docs_by_mal.get(key, []):
            stype = (doc.skjalategund or "").lower()
            utb = parse_date(doc.utbyting)
            is_question = ("fsp" in stype) or ("fyrirspurn" in stype)
            is_answer = ("svar" in stype) and not is_question
            if is_question and utb:
                if question_dt is None or utb.date() < question_dt:
                    question_dt = utb.date()
            if is_answer:
                attn_dt = prefer_athugasemd_date(athugasemd_by_skjalsnr.get(int(doc.skjalnr or -1)))
                cand = attn_dt or utb
                if cand:
                    if answer_dt is None or cand.date() < answer_dt:
                        answer_dt = cand.date()
        if question_dt is None and key in first_doc_date:
            question_dt = first_doc_date[key].date()
        status = ANSWER_STATUS_SVARAD if answer_dt else ANSWER_STATUS_OSVARAD
        latency = None
        if question_dt:
            latency = business_days_between(question_dt, answer_dt or dt.date.today())
        seen_keys.add(dedupe_key)
        metrics.append(metrics_model(
            lthing=lthing,
            malnr=key,
            answer_status=status,
            answer_latency=latency,
        ))
    return metrics


def parse_issue_documents(detail_xml: ET.Element, malflokkur: Optional[str] = None) -> List[Dict[str, Any]]:
    """Extract þingskjöl entries from a þingmál detail XML."""
    docs: List[Dict[str, Any]] = []

    def first_text(parent: ET.Element, tag: str) -> Optional[str]:
        for c in list(parent):
            if strip_ns(c.tag) == tag:
                return (c.text or "").strip()
        return None

    for child in detail_xml.iter():
        if strip_ns(child.tag) == "þingskjöl":
            for skjal in list(child):
                if strip_ns(skjal.tag) != "þingskjal":
                    continue
                try:
                    skjalnr = int(skjal.attrib.get("skjalsnúmer") or skjal.attrib.get("skjalsnumer"))
                except Exception:
                    skjalnr = None
                utbyting = first_text(skjal, "útbýting")
                skjalategund = first_text(skjal, "skjalategund")
                slod_html = None
                slod_pdf = None
                slod_xml = None
                for s in list(skjal):
                    if strip_ns(s.tag) == "slóð":
                        slod_html = first_text(s, "html") or slod_html
                        slod_pdf = first_text(s, "pdf") or slod_pdf
                        slod_xml = first_text(s, "xml") or slod_xml
                docs.append({
                    "malflokkur": malflokkur,
                    "skjalnr": skjalnr,
                    "utbyting": utbyting,
                    "skjalategund": skjalategund,
                    "slod_html": slod_html,
                    "slod_pdf": slod_pdf,
                    "slod_xml": slod_xml,
                })
    return docs


def parse_vote_details(vote_detail_xml: ET.Element) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary: Dict[str, Any] = {
        "ja": None,
        "nei": None,
        "greidirekki": None,
        "afgreidsla": None,
    }
    out: List[Dict[str, Any]] = []
    # samantekt block for counts
    for child in vote_detail_xml.iter():
        if strip_ns(child.tag) == "samantekt":
            for node in list(child):
                tag = strip_ns(node.tag)
                if tag == "afgreiðsla":
                    summary["afgreidsla"] = (node.text or "").strip()
                elif tag in {"já", "nei", "greiðirekkiatkvæði"}:
                    fjoldi_val = None
                    for g in node.iter():
                        if strip_ns(g.tag) == "fjöldi" and g.text:
                            try:
                                fjoldi_val = int(g.text)
                            except Exception:
                                fjoldi_val = None
                            break
                    if tag == "já":
                        summary["ja"] = fjoldi_val
                    elif tag == "nei":
                        summary["nei"] = fjoldi_val
                    else:
                        summary["greidirekki"] = fjoldi_val
    # per-voter breakdown
    for voter in vote_detail_xml.iter():
        if strip_ns(voter.tag) != "þingmaður":
            continue
        try:
            voter_id = int(voter.attrib.get("id")) if "id" in voter.attrib else None
        except Exception:
            voter_id = None
        name = None
        vote = None
        profile_xml = None
        for c in list(voter):
            tag = strip_ns(c.tag)
            if tag == "nafn":
                name = (c.text or "").strip()
            elif tag == "atkvæði":
                vote = (c.text or "").strip()
            elif tag == "xml":
                profile_xml = (c.text or "").strip()
        out.append({
            "voter_id": voter_id,
            "voter_name": name,
            "vote": vote,
            "voter_xml": profile_xml,
        })
    return summary, out


def speech_id_from_url(xml_url: Optional[str]) -> Optional[str]:
    if not xml_url:
        return None
    try:
        path = urlparse(xml_url).path
        if not path:
            return None
        return Path(path).stem or path.rsplit("/", 1)[-1]
    except Exception:
        return None


def member_id_from_url(nanar_url: Optional[str]) -> Optional[int]:
    if not nanar_url:
        return None
    try:
        qs = parse_qs(urlparse(nanar_url).query)
        if "nr" in qs and qs["nr"]:
            return int(qs["nr"][0])
    except Exception:
        return None
    return None


def speech_text_from_xml(detail_xml: ET.Element) -> str:
    text_node = None
    for node in detail_xml.iter():
        tag_plain = strip_ns(node.tag).lower()
        if tag_plain in ("ræðutexti", "raedutexti"):
            text_node = node
            break
    if text_node is None:
        return ""
    paras: List[str] = []
    for child in list(text_node):
        if not isinstance(child.tag, str):
            continue
        piece = " ".join(t.strip() for t in child.itertext() if (t or "").strip())
        if piece:
            paras.append(piece)
    if not paras:
        return " ".join(t.strip() for t in text_node.itertext() if (t or "").strip())
    return "\n".join(paras)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            text = data.strip()
            if text:
                self.parts.append(text)


def speech_text_from_html(content: bytes) -> str:
    parser = _TextExtractor()
    parser.feed(unescape(content.decode("utf-8", errors="replace")))
    return "\n".join(parser.parts)


def populate_speeches(
    session: Session,
    lthing: int,
    models_mod: Any,
    manual_models: Any,
    fetcher: Fetcher,
    max_records: Optional[int] = None,
) -> int:
    """
    Fetch all speeches for the given session and store normalized entries in manual_models.Speech.
    Relies on raedulisti__raeda table for metadata and downloads each speech XML for text + word counts.
    """
    if not manual_models or not hasattr(manual_models, "Speech"):
        return 0
    Speech = manual_models.Speech
    Raedulisti = getattr(models_mod, "RaedulistiRaeda")
    # Clear existing rows for this lthing to avoid stale data
    try:
        session.execute(text("DELETE FROM speech WHERE lthing=:lt"), {"lt": lthing})
        commit_with_retry(session)
    except Exception:
        session.rollback()
    speeches: List[Any] = []
    rows = session.execute(
        select(Raedulisti).where(Raedulisti.ingest_lthing == lthing)
    ).scalars().all()
    seen_speech_ids: set[tuple] = set()
    if max_records is not None:
        rows = rows[:max_records]
    for idx, row in enumerate(rows):
        if getattr(row, "leaf_raedumadur_forsetialthingis", None):
            # Skip speeches delivered in the role of President of Althingi.
            continue
        xml_url = getattr(row, "leaf_slodir_xml", None) or getattr(row, "leaf_slodir_html", None)
        speech_key = speech_id_from_url(xml_url) or getattr(row, "leaf_raedahofst", None) or f"speech-{idx}"
        if not xml_url:
            continue
        key_tuple = (lthing, speech_key)
        if key_tuple in seen_speech_ids:
            continue
        seen_speech_ids.add(key_tuple)
        try:
            content = fetcher.get(xml_url)
            detail_root = None
            speech_text = ""
            if xml_url.lower().endswith(".xml") or content.lstrip().startswith(b"<?xml"):
                detail_root = parse_xml(content, xml_url)
                speech_text = speech_text_from_xml(detail_root)
            else:
                speech_text = speech_text_from_html(content)
        except Exception as e:
            print(f"[warn] failed to fetch speech xml {xml_url}: {e}")
            speech_text = ""
            detail_root = None
        wc = len(re.findall(r"\w+", speech_text, flags=re.UNICODE)) if speech_text else 0
        start_dt = parse_date(getattr(row, "leaf_raedahofst", None))
        end_dt = parse_date(getattr(row, "leaf_raedulauk", None))
        duration_seconds = None
        if start_dt and end_dt:
            try:
                delta = end_dt - start_dt
                seconds = int(delta.total_seconds())
                duration_seconds = seconds if seconds >= 0 else None
            except Exception:
                duration_seconds = None
        wpm = None
        if duration_seconds and duration_seconds > 0 and wc:
            wpm = wc / (duration_seconds / 60.0)
        if wpm and wpm > 300:
            # Drop suspiciously high speeds that usually indicate bad timestamps.
            duration_seconds = None
            wpm = None
        m_id = member_id_from_url(getattr(row, "leaf_raedumadur_nanar", None))
        speeches.append(Speech(
            lthing=lthing,
            speech_id=speech_key,
            member_id=m_id,
            speaker_name=getattr(row, "leaf_raedumadur_nafn", None),
            speaker_role=getattr(row, "leaf_raedumadur_radherra", None) or getattr(row, "leaf_raedumadur_forsetialthingis", None) or getattr(row, "leaf_raedumadur_forsetiislands", None),
            date=getattr(row, "leaf_dagur", None),
            fundur=getattr(row, "leaf_fundur", None),
            fundarheiti=getattr(row, "leaf_fundarheiti", None),
            start_time=getattr(row, "leaf_raedahofst", None),
            end_time=getattr(row, "leaf_raedulauk", None),
            duration_seconds=duration_seconds,
            word_count=wc,
            words_per_minute=wpm,
            issue_malnr=getattr(row, "leaf_mal_malsnumer", None),
            issue_malsflokkur=getattr(row, "leaf_mal_malsflokkur", None),
            issue_malsheiti=getattr(row, "leaf_mal_malsheiti", None),
            kind=getattr(row, "leaf_tegundraedu", None),
            umraeda=getattr(row, "leaf_umraeda", None),
            audio_url=getattr(row, "leaf_slodir_hljod", None),
            html_url=getattr(row, "leaf_slodir_html", None),
            xml_url=xml_url,
            raw_text=speech_text,
        ))
        if len(speeches) % 200 == 0 and speeches:
            session.add_all(speeches)
            commit_with_retry(session)
            speeches = []
    if speeches:
        session.add_all(speeches)
        commit_with_retry(session)
    total = session.execute(text("SELECT COUNT(*) FROM speech WHERE lthing=:lt"), {"lt": lthing}).scalar_one_or_none() or 0
    print(f"[ok] speeches: stored {total} rows for lthing {lthing}")
    return total


def parse_member_seats(thingseta_xml: ET.Element, lthing: int) -> List[Dict[str, Any]]:
    seats: List[Dict[str, Any]] = []
    for ts in thingseta_xml.iter():
        if strip_ns(ts.tag) != "þingseta":
            continue
        thing = None
        for child in list(ts):
            if strip_ns(child.tag) == "þing" and child.text:
                try:
                    thing = int(child.text)
                except Exception:
                    thing = None
                break
        if thing != lthing:
            continue
        party = None
        party_id = None
        member_type = None
        inn = None
        ut = None
        kjordaemi = None
        kjordaemi_id = None
        for child in list(ts):
            tag = strip_ns(child.tag)
            if tag == "þingflokkur":
                party = (child.text or "").strip()
                try:
                    party_id = int(child.attrib.get("id")) if child.attrib.get("id") else None
                except Exception:
                    party_id = None
            elif tag == "tegund":
                member_type = (child.text or "").strip()
            elif tag == "kjördæmi":
                kjordaemi = (child.text or "").strip()
                try:
                    kjordaemi_id = int(child.attrib.get("id")) if child.attrib.get("id") else None
                except Exception:
                    kjordaemi_id = None
            elif tag == "tímabil":
                for tchild in list(child):
                    ttag = strip_ns(tchild.tag)
                    if ttag == "inn":
                        inn = (tchild.text or "").strip()
                    elif ttag == "út":
                        ut = (tchild.text or "").strip()
        seats.append({
            "party": party,
            "party_id": party_id,
            "type": member_type,
            "kjordaemi": kjordaemi,
            "kjordaemi_id": kjordaemi_id,
            "inn": inn,
            "ut": ut,
        })
    return seats


def parse_nefndarmenn(xml_root: ET.Element, nefnd_id: int, lthing: int) -> List[Dict[str, Any]]:
    members: List[Dict[str, Any]] = []

    def first_text(node: ET.Element, tag: str) -> Optional[str]:
        for c in list(node):
            if strip_ns(c.tag) == tag:
                return (c.text or "").strip()
        return None

    for nm in xml_root.iter():
        if strip_ns(nm.tag) not in {"nefndarmaður", "nefndarmadur"}:
            continue
        mid = None
        try:
            mid = int(nm.attrib.get("id")) if "id" in nm.attrib else None
        except Exception:
            mid = None
        name = first_text(nm, "nafn")
        role = first_text(nm, "hlutverk") or first_text(nm, "tegund")
        inn = first_text(nm, "inn")
        ut = first_text(nm, "út") or first_text(nm, "ut")
        members.append({
            "member_id": mid,
            "name": name,
            "role": role,
            "inn": inn,
            "ut": ut,
            "nefnd_id": nefnd_id,
            "lthing": lthing,
        })
    return members


def parse_nefndarmenn_all(xml_root: ET.Element, lthing: int) -> Dict[int, List[Dict[str, Any]]]:
    """Parse the aggregated nefndarmenn feed which contains all committees."""
    def first_text(node: ET.Element, tag: str) -> Optional[str]:
        for c in list(node):
            if strip_ns(c.tag) == tag:
                return (c.text or "").strip()
        return None

    per_nefnd: Dict[int, List[Dict[str, Any]]] = {}
    for nefnd in xml_root.iter():
        if strip_ns(nefnd.tag) != "nefnd":
            continue
        nefnd_id_raw = nefnd.attrib.get("id")
        try:
            nefnd_id = int(nefnd_id_raw) if nefnd_id_raw is not None else None
        except Exception:
            nefnd_id = None
        if nefnd_id is None:
            continue
        members: List[Dict[str, Any]] = []
        seen_keys: set[Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]] = set()
        for nm in list(nefnd):
            if strip_ns(nm.tag) not in {"nefndarmaður", "nefndarmadur"}:
                continue
            try:
                mid = int(nm.attrib.get("id")) if "id" in nm.attrib else None
            except Exception:
                mid = None
            name = first_text(nm, "nafn")
            role = first_text(nm, "staða") or first_text(nm, "hlutverk") or first_text(nm, "tegund")
            inn = first_text(nm, "nefndasetahófst") or first_text(nm, "inn")
            ut = first_text(nm, "nefndasetulauk") or first_text(nm, "út") or first_text(nm, "ut")
            dedupe_key = (mid, name, inn, ut)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            members.append({
                "member_id": mid,
                "name": name,
                "role": role,
                "inn": inn,
                "ut": ut,
                "nefnd_id": nefnd_id,
                "lthing": lthing,
            })
        per_nefnd[nefnd_id] = members
    return per_nefnd


# -----------------------------
# Main ingestion
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="althingi.db", help="SQLite DB path")
    ap.add_argument("--reset-db", action=argparse.BooleanOptionalAction, default=False, help="Delete existing DB file before ingest (default: False)")
    ap.add_argument("--schema", default="schema_map.json", help="schema_map.json generated by check_data.py")
    ap.add_argument("--models-dir", default="app", help="Directory containing generated models.py")
    ap.add_argument("--max-records", type=int, default=None, help="Limit records per resource (debug)")
    ap.add_argument("--sleep", type=float, default=0.15, help="Sleep between requests")
    ap.add_argument("--cache-dir", default="data/cache", help="Directory for HTTP cache (default: data/cache)")
    ap.add_argument("--force-fetch", action="store_true", help="Ignore cache and re-download all resources")
    ap.add_argument("--speeches-only", action="store_true", help="Only refresh raedur (skip other resources; keeps existing DB)")
    ap.add_argument("--skip-speeches", action="store_true", help="Skip fetching individual raedur XML documents")
    ap.add_argument("--lthing", type=int, default=None, help="Fetch a specific löggjafarþing number")
    ap.add_argument("--lthing-range", type=str, default=None, help="Inclusive range start,end (e.g. 122,145)")
    ap.add_argument("--all-lthing", action="store_true", help="Fetch all löggjafarþing sessions (uses existing DB)")
    args = ap.parse_args()

    # Load schema map
    with open(args.schema, "r", encoding="utf-8") as f:
        schema_map = json.load(f)

    resources = schema_map.get("resources", [])
    range_targets: List[int] = []
    if args.lthing_range:
        try:
            start_s, end_s = args.lthing_range.split(",", 1)
            start_i = int(start_s.strip())
            end_i = int(end_s.strip())
            lo, hi = sorted((start_i, end_i))
            range_targets = list(range(lo, hi + 1))
        except Exception as e:
            raise SystemExit(f"Invalid --lthing-range '{args.lthing_range}': {e}") from e

    if args.speeches_only:
        resources = [r for r in resources if r.get("table") == "raedulisti__raeda"]
        if not resources:
            raise RuntimeError("schema_map is missing raedulisti resource required for --speeches-only")

    if args.speeches_only:
        args.reset_db = False
    if args.all_lthing:
        args.reset_db = False
    if args.lthing is not None:
        args.reset_db = False

    # Ensure DB directory exists and reset if requested
    db_path = os.path.abspath(args.db)
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    if args.reset_db and os.path.exists(db_path):
        os.remove(db_path)

    # Import generated models.py
    models_dir = os.path.abspath(args.models_dir)
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    models = importlib.import_module("models")
    Base = getattr(models, "Base")
    # Register manual models (IssueDocument)
    try:
        manual_models = importlib.import_module("manual_models")
    except ModuleNotFoundError:
        manual_models = None

    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"timeout": 60})

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA busy_timeout=60000")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
        except Exception:
            pass
    Base.metadata.create_all(engine)
    if manual_models and hasattr(manual_models, "Base"):
        manual_models.Base.metadata.create_all(engine)

    base_cache_dir = Path(args.cache_dir) if args.cache_dir else None
    fetcher = Fetcher(sleep_s=args.sleep, cache_dir=str(base_cache_dir) if base_cache_dir else None, force=args.force_fetch)
    detail_cache: Dict[str, List[Tuple[int, str]]] = {}
    fetched_at = now_utc_iso()
    current_lthing_now, _ = discover_current_lthing_and_yfirlit(fetcher)

    def resource_urls_for_lthing(base_resources: List[Dict[str, Any]], lthing_val: int, yfirlit_urls: Dict[str, str]) -> List[Dict[str, Any]]:
        out = []
        for r in base_resources:
            rcopy = dict(r)
            name = rcopy.get("name")
            if name and name in yfirlit_urls:
                rcopy["url"] = yfirlit_urls[name]
            else:
                url = rcopy.get("url", "")
                if "lthing=" in url:
                    rcopy["url"] = re.sub(r"(lthing=)\\d+", rf"\\g<1>{lthing_val}", url)
            out.append(rcopy)
        return out

    if args.all_lthing:
        sessions = list_lthing_sessions(fetcher)
        targets = [(n, y) for n, y in sessions if y]
    elif range_targets:
        targets = []
        for lt_val in range_targets:
            try:
                lt, y = discover_lthing_and_yfirlit(fetcher, lt_val)
                targets.append((lt, y))
            except Exception as e:
                print(f"[warn] skipping lthing {lt_val}: {e}")
    elif args.lthing is not None:
        lt, y = discover_lthing_and_yfirlit(fetcher, args.lthing)
        targets = [(lt, y)]
    else:
        lt, y = discover_current_lthing_and_yfirlit(fetcher)
        targets = [(lt, y)]

    # Build a lookup of tablename -> ModelClass
    model_by_table: Dict[str, Any] = {}
    for cls in Base.registry.mappers:
        mapped = cls.class_
        tab = getattr(mapped, "__tablename__", None)
        if tab:
            model_by_table[tab] = mapped

    with Session(engine) as session:
        issue_documents: List[Any] = []
        vote_details_to_add: List[Any] = []
        seats_to_add: List[Any] = []
        for lthing, yfirlit in targets:
            print(f"[info] fetching lthing {lthing}")
            fetcher.cache_only_default = (lthing != current_lthing_now and not args.force_fetch)
            if base_cache_dir:
                cache_dir_for_lt = base_cache_dir / str(lthing)
                cache_dir_for_lt.mkdir(parents=True, exist_ok=True)
                fetcher.cache_dir = str(cache_dir_for_lt)
            scoped_resources = resource_urls_for_lthing(resources, lthing, yfirlit)
            for r in scoped_resources:
                if "error" in r:
                    continue
                name = r["name"]
                url = r["url"]
                table = r["table"]
                record_path = r["record_path"]
    
                Model = model_by_table.get(table)
                if Model is None:
                    raise RuntimeError(f"Model for table '{table}' not found. Re-run check_data.py?")
    
                # Fetch resource XML
                xb = fetcher.get(url)
                root = parse_xml(xb, url)
                records = iter_records(root, record_path)
                if args.max_records is not None:
                    records = records[:args.max_records]
    
                # Refresh this resource for this session
                execute_with_retry(session, text(f'DELETE FROM "{table}" WHERE ingest_lthing=:lt AND ingest_resource=:res'),
                                   {"lt": lthing, "res": name})

                # Also clear child tables
                for ct in r.get("child_tables", []):
                    ctable = ct["table"]
                    execute_with_retry(session, text(f'DELETE FROM "{ctable}" WHERE ingest_lthing=:lt AND ingest_resource=:res'),
                                       {"lt": lthing, "res": name})
    
                session.flush()
    
                attr_map: Dict[str, str] = r["attr_map"]
                leaf_map: Dict[str, str] = r["leaf_map"]
                repeated_paths: set[str] = set(r.get("repeated_leaf_paths", []))
    
                child_by_path: Dict[str, str] = {ct["path"]: ct["table"] for ct in r.get("child_tables", [])}
                child_models: Dict[str, Any] = {ct["table"]: model_by_table[ct["table"]] for ct in r.get("child_tables", [])}
                unique_constraints = unique_constraint_columns(Model)
                seen_unique: List[set] = [set() for _ in unique_constraints]
    
                parents: List[Any] = []
                children_to_add: List[Any] = []
                skipped_dupes = 0
    
                for rec in records:
                    row: Dict[str, Any] = {}
                    row["ingest_lthing"] = lthing
                    row["ingest_resource"] = name
                    row["source_url"] = url
                    row["fetched_at"] = fetched_at
                    row["raw_xml"] = ET.tostring(rec, encoding="unicode")
    
                    # attributes
                    for k, col in attr_map.items():
                        if k in rec.attrib:
                            row[col] = rec.attrib.get(k)
    
                    # leaf fields
                    leafs = iter_leaf_fields(rec)
                    # group leaf values per path
                    per_path: Dict[str, List[str]] = {}
                    for p, v in leafs:
                        per_path.setdefault(p, []).append(v)
    
                    # scalar leaves -> first value
                    for p, col in leaf_map.items():
                        vals = per_path.get(p)
                        if vals:
                            row[col] = vals[0]
    
                    # Resource-specific enrichment: fetch flutningsmenn from detailed thingskjal XML
                    if name == "þingskjalalisti":
                        detail_url = row.get("leaf_slod_xml")
                        if detail_url:
                            if detail_url not in detail_cache:
                                try:
                                    detail_xml = parse_xml(fetcher.get(detail_url), detail_url)
                                    detail_cache[detail_url] = parse_flutningsmenn(detail_xml)
                                except Exception:
                                    detail_cache[detail_url] = []
                            fms = detail_cache.get(detail_url) or []
                            if fms:
                                row["leaf_kalladaftur"] = json.dumps(fms, ensure_ascii=False)
    
                    # Drop duplicates within this batch based on UniqueConstraints (if any)
                    is_duplicate = False
                    for idx, cols in enumerate(unique_constraints):
                        key = tuple(row.get(c) for c in cols)
                        if any(v is None for v in key):
                            continue
                        if key in seen_unique[idx]:
                            is_duplicate = True
                            break
                        seen_unique[idx].add(key)
                    if is_duplicate:
                        skipped_dupes += 1
                        continue
    
                    parent = Model(**_filter_model_kwargs(Model, row))
                    parents.append(parent)
                    session.add(parent)
    
                    # repeated leaves -> child tables
                    for p in repeated_paths:
                        vals = per_path.get(p) or []
                        if not vals:
                            continue
                        ctable = child_by_path.get(p)
                        if not ctable:
                            continue
                        ChildModel = child_models[ctable]
                        # parent.id available after flush; do later
                        # store temporary tuple
                        children_to_add.append((parent, ChildModel, vals, url, fetched_at, lthing, name))
    
                session.flush()
    
                # Now insert child rows (needs parent.id)
                for (parent, ChildModel, vals, url, fetched_at, lt, resname) in children_to_add:
                    for i, v in enumerate(vals, start=1):
                        session.add(ChildModel(
                            parent_id=parent.id,
                            seq=i,
                            value=v,
                            ingest_lthing=lt,
                            ingest_resource=resname,
                            source_url=url,
                            fetched_at=fetched_at
                        ))
    
                session.flush()
                # Extra: fetch and store per-voter breakdown for atkvæðagreiðslur
                if name == "atkvæðagreiðslur" and manual_models and hasattr(manual_models, "VoteDetail"):
                    VoteDetail = manual_models.VoteDetail
                    # clear existing for this lthing
                    session.execute(text('DELETE FROM vote_details WHERE lthing=:lt'), {"lt": lthing})
                    session.flush()
                    # attr map name for vote number
                    vote_num_col = attr_map.get("atkvæðagreiðslunúmer")
                    nanar_col = leaf_map.get("nánar/xml") or leaf_map.get("nánar")
                    ja_col = leaf_map.get("samantekt/já/fjöldi")
                    nei_col = leaf_map.get("samantekt/nei/fjöldi")
                    greidirekki_col = leaf_map.get("samantekt/greiðirekkiatkvæði/fjöldi")
                    afgreidsla_col = leaf_map.get("samantekt/afgreiðsla")
                    for parent in parents:
                        detail_url = getattr(parent, nanar_col, None) if nanar_col else None
                        vote_num = getattr(parent, vote_num_col, None) if vote_num_col else None
                        if not detail_url or vote_num is None:
                            continue
                        try:
                            vxml = parse_xml(fetcher.get(detail_url), detail_url)
                            summary_counts, voter_rows = parse_vote_details(vxml)
                            # update parent summary fields
                            if ja_col:
                                setattr(parent, ja_col, summary_counts.get("ja"))
                            if nei_col:
                                setattr(parent, nei_col, summary_counts.get("nei"))
                            if greidirekki_col:
                                setattr(parent, greidirekki_col, summary_counts.get("greidirekki"))
                            if afgreidsla_col:
                                setattr(parent, afgreidsla_col, summary_counts.get("afgreidsla"))
                            for vr in voter_rows:
                                vote_details_to_add.append(VoteDetail(
                                    lthing=lthing,
                                    vote_num=vote_num,
                                    parent_id=parent.id,
                                    voter_id=vr.get("voter_id"),
                                    voter_name=vr.get("voter_name"),
                                    voter_xml=vr.get("voter_xml"),
                                    vote=vr.get("vote"),
                                ))
                        except Exception as e:
                            print(f"[warn] failed to fetch vote breakdown for {vote_num}: {e}")
    
                # Extra: fetch and store member seats (party/type) for þingmenn
                if name == "þingmannalisti" and manual_models and hasattr(manual_models, "MemberSeat"):
                    MemberSeat = manual_models.MemberSeat
                    execute_with_retry(session, text('DELETE FROM member_seat WHERE lthing=:lt'), {"lt": lthing})
                    session.flush()
                    thingseta_col = leaf_map.get("xml/þingseta")
                    for parent in parents:
                        thingseta_url = getattr(parent, thingseta_col, None) if thingseta_col else None
                        member_id = getattr(parent, attr_map.get("id"), None)
                        if not thingseta_url or member_id is None:
                            continue
                        try:
                            ts_xml = parse_xml(fetcher.get(thingseta_url), thingseta_url)
                            seats = parse_member_seats(ts_xml, lthing)
                            seen_keys = set()
                            for s in seats:
                                key = (member_id, s.get("party_id"), s.get("inn") or "")
                                if key in seen_keys:
                                    continue
                                seen_keys.add(key)
                                seats_to_add.append(MemberSeat(
                                    lthing=lthing,
                                    member_id=member_id,
                                    party_id=s.get("party_id"),
                                    party_name=s.get("party"),
                                    type=s.get("type"),
                                    kjordaemi_id=s.get("kjordaemi_id"),
                                    kjordaemi_name=s.get("kjordaemi"),
                                    inn=s.get("inn"),
                                    ut=s.get("ut"),
                                ))
                        except Exception as e:
                            print(f"[warn] failed to fetch þingseta for member {member_id}: {e}")
    
                # Extra: fetch nefndarmenn for each nefnd
                if name == "nefndir" and manual_models and hasattr(manual_models, "NefndMember"):
                    NefndMember = manual_models.NefndMember
                    execute_with_retry(session, text('DELETE FROM nefnd_member WHERE lthing=:lt'), {"lt": lthing})
                    session.flush()
                    nefndarmenn_url = None
                    if parents and leaf_map:
                        candidate = getattr(parents[0], leaf_map.get("nefndarmenn"), None)
                        if candidate:
                            nefndarmenn_url = candidate
                    if nefndarmenn_url and "nnefnd=" in nefndarmenn_url:
                        nefndarmenn_url = nefndarmenn_url.replace("nnefnd=", "nefnd=")
                    if nefndarmenn_url:
                        nefndarmenn_url = norm_url(url, nefndarmenn_url)
                    if not nefndarmenn_url:
                        print("[warn] no nefndarmenn url found")
                    else:
                        try:
                            xml_root = parse_xml(fetcher.get(nefndarmenn_url), nefndarmenn_url)
                            parsed_map = parse_nefndarmenn_all(xml_root, lthing)
                            nefnd_rows = []
                            seen_nm_keys: set[Tuple[int, int, Optional[int], Optional[str], Optional[str]]] = set()
                            for parent in parents:
                                nefnd_id = getattr(parent, r["attr_map"].get("id"), None) if r.get("attr_map") else None
                                try:
                                    nefnd_id_int = int(nefnd_id) if nefnd_id is not None else None
                                except Exception:
                                    nefnd_id_int = None
                                if nefnd_id_int is None:
                                    continue
                                entries = parsed_map.get(nefnd_id_int, [])
                                if entries:
                                    for p in entries:
                                        key = (lthing, nefnd_id_int, p.get("member_id"), p.get("name"), p.get("inn"))
                                        if key in seen_nm_keys:
                                            continue
                                        seen_nm_keys.add(key)
                                        nefnd_rows.append(NefndMember(**p))
                                    print(f"[ok] nefnd {nefnd_id_int}: stored {len(entries)} nefndarmenn")
                            if nefnd_rows:
                                session.add_all(nefnd_rows)
                                commit_with_retry(session)
                        except Exception as e:
                            print(f"[warn] failed to fetch aggregated nefndarmenn: {e}")
    
                commit_with_retry(session)
                inserted = len(records) - skipped_dupes
                print(f"[ok] {name}: inserted {inserted} rows into {table} ({skipped_dupes} skipped as duplicates)")
                # If we just processed þingmálalisti, collect issue documents from detail XMLs
                if name == "þingmálalisti" and manual_models and hasattr(manual_models, "IssueDocument"):
                    issue_documents = []
                    IssueDocModel = manual_models.IssueDocument
                    execute_with_retry(session, text('DELETE FROM issue_documents WHERE lthing=:lt'), {"lt": lthing})
                    session.flush()
                    seen_issue_docs: set[Tuple[int, int, Optional[int]]] = set()
                    for parent in parents:
                        detail_url = getattr(parent, r["leaf_map"].get("xml"), None) if r.get("leaf_map") else None
                        malnr = getattr(parent, r["attr_map"].get("málsnúmer"), None) if r.get("attr_map") else None
                        if not detail_url or malnr is None:
                            continue
                        try:
                            detail_xml = parse_xml(fetcher.get(detail_url), detail_url)
                            docs = parse_issue_documents(
                                detail_xml,
                                getattr(parent, r["attr_map"].get("málsflokkur"), None) if r.get("attr_map") else None,
                            )
                            for d in docs:
                                skjalnr = d.get("skjalnr")
                                key = (lthing, int(malnr), skjalnr if skjalnr is None else int(skjalnr))
                                if key in seen_issue_docs:
                                    continue
                                seen_issue_docs.add(key)
                                issue_documents.append(IssueDocModel(
                                    lthing=lthing,
                                    malnr=malnr,
                                    malflokkur=d.get("malflokkur"),
                                    skjalnr=skjalnr,
                                    skjalategund=d.get("skjalategund"),
                                    utbyting=d.get("utbyting"),
                                    slod_html=d.get("slod_html"),
                                    slod_pdf=d.get("slod_pdf"),
                                    slod_xml=d.get("slod_xml"),
                                ))
                        except Exception as e:
                            print(f"[warn] failed to fetch þingskjöl for mál {malnr}: {e}")
                    if issue_documents:
                        session.add_all(issue_documents)
                        commit_with_retry(session)
                        print(f"[ok] þingmálalisti: stored {len(issue_documents)} issue_documents")
                if name == "atkvæðagreiðslur" and vote_details_to_add:
                    session.add_all(vote_details_to_add)
                    commit_with_retry(session)
                    print(f"[ok] atkvæðagreiðslur: stored {len(vote_details_to_add)} vote_details")
                    vote_details_to_add.clear()

                if name == "þingmannalisti" and seats_to_add:
                    session.add_all(seats_to_add)
                    commit_with_retry(session)
                    print(f"[ok] þingmannalisti: stored {len(seats_to_add)} member seats")
                    seats_to_add.clear()

    
    # Extra: cache nefndarfundir + fundargerðir for this þing to support attendance parsing
    if args.cache_dir and not args.speeches_only:
        for lthing, _ in targets:
            try:
                cache_nefndarfundir(fetcher, lthing)
            except Exception as e:
                print(f"[warn] failed to cache nefndarfundir: {e}")

    # Aggregate vote sessions and committee attendance into manual tables
    if manual_models:
        with Session(engine) as session:
            for lthing, _ in targets:
                # map abbrev -> member_id
                abbr_map = {}
                if not args.speeches_only:
                    people = session.execute(
                        select(models.ThingmannalistiThingmadur).where(models.ThingmannalistiThingmadur.ingest_lthing == lthing)
                    ).scalars().all()
                    for p in people:
                        if p.attr_id is not None and p.leaf_skammstofun:
                            abbr_map[_norm_tag(p.leaf_skammstofun)] = int(p.attr_id)
                    # derived issue metrics (answer status/latency) for written questions
                    if hasattr(manual_models, "IssueMetrics") and hasattr(manual_models, "IssueDocument"):
                        try:
                            execute_with_retry(session, text("DELETE FROM issue_metrics WHERE lthing=:lt"), {"lt": lthing})
                            issue_rows = session.execute(
                                select(models.ThingmalalistiMal).where(models.ThingmalalistiMal.ingest_lthing == lthing)
                            ).scalars().all()
                            issue_docs = session.execute(
                                select(manual_models.IssueDocument).where(manual_models.IssueDocument.lthing == lthing)
                            ).scalars().all()
                            ath_map: Dict[int, str] = {}
                            for nr, ath in session.execute(
                                select(models.ThingskjalalistiThingskjal.attr_skjalsnumer, models.ThingskjalalistiThingskjal.leaf_athugasemd)
                                .where(models.ThingskjalalistiThingskjal.ingest_lthing == lthing)
                            ).all():
                                if nr is None or not ath:
                                    continue
                                try:
                                    ath_map[int(nr)] = ath
                                except Exception:
                                    continue
                            metrics = compute_issue_metrics(lthing, issue_docs, issue_rows, ath_map, manual_models)
                            if metrics:
                                session.add_all(metrics)
                            commit_with_retry(session)
                            print(f"[ok] issue_metrics: stored {len(metrics)} rows")
                        except Exception as e:
                            session.rollback()
                            print(f"[warn] issue_metrics failed: {e}")
                    # vote sessions
                    populate_vote_sessions_and_attendance(session, lthing, manual_models)
                    # committee attendance
                    cache_dir = Path(args.cache_dir)
                    populate_committee_attendance(session, cache_dir, lthing, abbr_map, manual_models)
                if not args.skip_speeches:
                    try:
                        populate_speeches(session, lthing, models, manual_models, fetcher, max_records=args.max_records)
                    except Exception as e:
                        print(f"[warn] populate_speeches failed: {e}")

    print(f"Done. DB: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
