from __future__ import annotations

import datetime as dt
import re
import unicodedata
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Blueprint, current_app, render_template, request
from sqlalchemy import select, func, text, or_
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from urllib.parse import quote

from . import models
try:
    import app.manual_models as manual_models  # type: ignore
except Exception:
    manual_models = None
from .views_helper import (
    current_lthing,
    icelandic_sort_key,
    flutningsmenn_primary_id,
)
from .utils.dates import (
    business_days_between,
    parse_date,
    prefer_athugasemd_date,
)
from .manual_models import IssueDocument, VoteDetail, MemberSeat, NefndMember, attach_flutningsmenn
from .constants import WRITTEN_QUESTION_LABEL, ANSWER_STATUS_SVARAD, ANSWER_STATUS_OSVARAD, canonical_issue_type

bp = Blueprint("main", __name__)


def _selected_lthing(session: Session) -> Optional[int]:
    return request.args.get("lthing", type=int) or current_lthing(session)


def _get_engine():
    engine = current_app.config.get("ENGINE")
    if engine is None:
        raise RuntimeError("Database engine is not configured on the Flask app")
    return engine


def _cache_dir() -> Path:
    """Return the ingestion cache directory (where get_data stores fetched XML)."""
    return Path(current_app.root_path).parent / "data" / "cache"


def _norm_tag(tag: str) -> str:
    """Normalize an XML tag to ASCII for matching (strip accents/punctuation)."""
    trans = {
        0xf0: "d",  # ð
        0xd0: "D",  # Ð
        0xfe: "th",  # þ
        0xde: "Th",  # Þ
        0xe6: "ae",  # æ
        0xc6: "Ae",  # Æ
        0xf6: "o",   # ö
        0xd6: "O",   # Ö
        0xe1: "a",   # á
        0xc1: "A",   # Á
        0xe9: "e",   # é
        0xc9: "E",   # É
        0xed: "i",   # í
        0xcd: "I",   # Í
        0xf3: "o",   # ó
        0xd3: "O",   # Ó
        0xfa: "u",   # ú
        0xda: "U",   # Ú
        0xfd: "y",   # ý
        0xdd: "Y",   # Ý
    }
    cleaned = tag.translate(trans)
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum())
    return cleaned.lower()


def _strip_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _cache_path_for_url(url: str, cache_dir: Path) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", url)
    path = cache_dir / safe
    if not path.exists() and "/thingmal/?" in url:
        alt_url = url.replace("/thingmal/?", "/thingmal/")
        alt_safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", alt_url)
        path = cache_dir / alt_safe
    return path


def _cached_issue_meta(url: Optional[str], cache_dir: Path, cache: Dict[str, tuple]) -> tuple[Optional[str], Optional[int]]:
    """
    Read status text and count of umsagnir for an issue from its cached detail XML.
    Returns (status, umsagn_count) or (None, None) if unavailable.
    """
    if not url:
        return None, None
    if url in cache:
        return cache[url]
    path = _cache_path_for_url(url, cache_dir)
    status = None
    umsagn_count: Optional[int] = None
    if path.exists():
        try:
            root = ET.parse(path).getroot()
            # status text (anywhere in doc)
            for node in root.iter():
                if _norm_tag(node.tag) == "staamals":
                    status = (node.text or "").strip() or None
                    break
            # count erindi (actual submissions) anywhere in doc
            erindi_nums = set()
            for u in root.iter():
                if _norm_tag(u.tag) == "erindi":
                    num = None
                    for k, v in u.attrib.items():
                        if _norm_tag(k) == "dagbokarnumer":
                            num = v
                            break
                    if num is None:
                        num = (
                            u.attrib.get("dagbókarnúmer")
                            or u.attrib.get("dagbokarnumer")
                            or u.attrib.get("dagbókarnumer")
                        )
                    erindi_nums.add(num or "unknown")
            umsagn_count = len(erindi_nums)
        except Exception:
            status = None
            umsagn_count = None
    cache[url] = (status, umsagn_count)
    return cache[url]


def _parse_iso(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(ts)
    except Exception:
        return None
    return None


def _in_intervals(ts: Optional[dt.datetime], intervals: List[Tuple[Optional[dt.datetime], Optional[dt.datetime]]]) -> bool:
    if ts is None:
        return False
    ts_date = ts.date()
    for inn, ut in intervals:
        if inn:
            inn_date = inn.date() if isinstance(inn, dt.datetime) else inn
            if ts_date < inn_date:
                continue
        if ut:
            ut_date = ut.date() if isinstance(ut, dt.datetime) else ut
            if ts_date > ut_date:
                continue
        return True
    return False


def _effective_intervals(seats: List[Any]) -> Dict[int, List[Tuple[Optional[dt.datetime], Optional[dt.datetime]]]]:
    """
    Build intervals per member_id where they are actually seated (ignore varamenn).
    Only include seats where type is not 'varamaður'.
    """
    out: Dict[int, List[Tuple[Optional[dt.datetime], Optional[dt.datetime]]]] = defaultdict(list)
    for seat in seats:
        mid = getattr(seat, "member_id", None)
        if mid is None:
            continue
        seat_type = str(getattr(seat, "type", "") or "").lower()
        if seat_type.strip() == "varamaður":
            continue
        inn_dt = parse_date(seat.inn)
        ut_dt = parse_date(seat.ut)
        try:
            out[int(mid)].append((inn_dt, ut_dt))
        except Exception:
            continue
    return out


def _fmt_duration(start: Optional[dt.datetime], end: Optional[dt.datetime]) -> Optional[str]:
    if not start or not end:
        return None
    delta = end - start
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return None
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _fmt_seconds(seconds: Optional[int]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        sec = int(seconds)
    except Exception:
        return None
    if sec < 0:
        return None
    minutes, sec = divmod(sec, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _pseudo_member_id(name: str) -> Optional[int]:
    key = (name or "").strip()
    if not key:
        return None
    value = zlib.crc32(key.casefold().encode("utf-8")) or 1
    return -int(value)


def _minister_records(session: Session, lthing: Optional[int]) -> List[Dict[str, Any]]:
    Speech = getattr(manual_models, "Speech", None) if manual_models else None
    if not Speech:
        return []
    filters = [
        Speech.member_id.is_(None),
        Speech.speaker_name.is_not(None),
        Speech.speaker_role.is_not(None),
    ]
    if lthing is not None:
        filters.append(Speech.lthing == lthing)
    role_lower = func.lower(Speech.speaker_role)
    role_filter = or_(role_lower.like("%ráðherra%"), role_lower.like("%radherra%"))
    rows = session.execute(
        select(
            Speech.speaker_name,
            func.max(Speech.speaker_role),
            func.count(),
        ).where(*filters, role_filter).group_by(Speech.speaker_name)
    ).all()
    out = []
    for name, role, count in rows:
        if not name:
            continue
        out.append({
            "name": name,
            "role": role,
            "count": count or 0,
        })
    return out


def _minister_map(session: Session, lthing: Optional[int]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in _minister_records(session, lthing):
        pid = _pseudo_member_id(rec["name"])
        if pid is None:
            continue
        out[rec["name"].strip()] = {
            "id": pid,
            "role": rec.get("role"),
            "count": rec.get("count", 0),
        }
    return out


def _cached_speeches(
    url: Optional[str],
    cache_dir: Path,
    cache: Dict[str, List[Dict[str, Any]]],
    party_by_member: Optional[Dict[int, str]] = None,
    name_to_member_id: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Parse speeches for an issue from cached detail XML.
    Returns list of dicts with possible nested andsvör.
    """
    if not url:
        return []
    if url in cache:
        return cache[url]
    path = _cache_path_for_url(url, cache_dir)
    speeches: List[Dict[str, Any]] = []
    if not path.exists():
        cache[url] = speeches
        return speeches
    try:
        root = ET.parse(path).getroot()
        # speeches are direct children of <ræður>; keep document order
        raedur = root.find(".//ræður")
        nodes = list(raedur) if raedur is not None else []
        if not nodes:
            nodes = [n for n in root if _norm_tag(n.tag) == "ræða"]
        base = None
        for r in nodes:
            speaker = None
            speaker_id = None
            is_forseti = False
            radherra_text = None
            # pick first <nafn> anywhere under the speech
            for c in r.iter():
                tag = _strip_tag(c.tag)
                if tag == "nafn":
                    speaker = (c.text or "").strip() or None
                    speaker_id = c.attrib.get("id") or c.attrib.get("ID")
                    if speaker:
                        break
                if tag == "forsetiAlþingis":
                    is_forseti = True
                if tag == "ráðherra":
                    radherra_text = (c.text or "").strip() or None
            if not speaker and speaker_id:
                speaker = f"Þingmaður {speaker_id}"
            if not speaker:
                # fallback to ráðherra text if present
                for c in r.iter():
                    if _strip_tag(c.tag) == "ráðherra":
                        speaker = (c.text or "").strip() or None
                        break
            if not speaker:
                # final fallback: use þingmaður attribute if present
                tm_id = None
                for k, v in r.attrib.items():
                    if _strip_tag(k) in ("þingmaður", "thingmadur", "thingmaður"):
                        tm_id = v
                        break
                if tm_id:
                    speaker = f"Þingmaður {tm_id}"
            if not speaker and is_forseti:
                speaker = "Forseti Alþingis"
            if not speaker_id and name_to_member_id and speaker:
                speaker_id = name_to_member_id.get(speaker.strip())
            start = None
            end = None
            kind = None
            html_link = None
            for c in r.iter():
                tag = _strip_tag(c.tag)
                if tag == "ræðahófst" and start is None:
                    start = _parse_iso((c.text or "").strip())
                elif tag == "ræðulauk" and end is None:
                    end = _parse_iso((c.text or "").strip())
                elif tag == "tegundræðu" and kind is None:
                    kind = (c.text or "").strip()
                elif tag == "html" and html_link is None:
                    html_link = (c.text or "").strip() or None
            duration = _fmt_duration(start, end)
            party = None
            if speaker_id and party_by_member:
                try:
                    party = party_by_member.get(int(speaker_id))
                except Exception:
                    party = None
            if not party and radherra_text:
                party = radherra_text
            raeda_dict = {
                "speaker": speaker or "Ónafngreindur",
                "speaker_id": speaker_id,
                "start": start,
                "end": end,
                "duration": duration,
                "kind": kind,
                "html": html_link,
                "party": party,
                "role": radherra_text,
                "andsvor": [],
            }
            is_andsvar = (kind or "").strip().lower() == "andsvar"
            if is_andsvar and base is not None:
                base["andsvor"].append(raeda_dict)
            else:
                speeches.append(raeda_dict)
                base = raeda_dict
    except Exception:
        speeches = []
    cache[url] = speeches
    return speeches


def _norm_abbr(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if ch.isalnum())
    return s.lower()


def _parse_attendance_from_html(html_text: str, abbr_to_member: Dict[str, int]) -> Tuple[set, set]:
    """
    Return (attended_ids, absent_ids) from HTML-ish texti content.
    Rules:
    - Lines with 'fyrir' credit attendance to the target after 'fyrir'
    - Lines with 'fjarverandi' or 'boðaði forföll' count as absence for that member
    - Other 'Name (ABBR)' lines count as attendance for ABBR
    """
    attended: set = set()
    absent: set = set()
    if not html_text:
        return attended, absent
    # Extract section after "Mætt:" until next <h2 or end
    lower = html_text.lower()
    start = lower.find("mætt:")
    if start == -1:
        return attended, absent
    section = html_text[start:]
    # Stop at next heading marker
    stop_markers = ["<h2", "bókað:", "b\u00f3ka\u00f0:"]
    stop = len(section)
    for m in stop_markers:
        idx = section.lower().find(m)
        if idx != -1:
            stop = min(stop, idx)
    section = section[:stop]
    parts = re.split(r"<br\\s*/?>", section, flags=re.IGNORECASE)
    for raw in parts:
        line = raw.strip()
        if not line:
            continue
        lower_line = line.lower()
        # absence markers
        if "fjarverandi" in lower_line or "boðaði forföll" in lower_line or "bo\u00f0a\u00f0i forf\u00f6ll" in lower_line:
            m = re.search(r"\(([^\)]+)\)", line)
            if m:
                abbr = _norm_abbr(m.group(1))
                mid = abbr_to_member.get(abbr)
                if mid:
                    absent.add(mid)
            continue
        # proxy: X (...) fyrir Y (...)
        if "fyrir" in lower_line:
            target = None
            m = re.findall(r"\(([^\)]+)\)", line)
            if len(m) >= 2:
                target_abbr = _norm_abbr(m[-1])
                target = abbr_to_member.get(target_abbr)
            if target:
                attended.add(target)
            continue
        # normal attendance
        m = re.search(r"\(([^\)]+)\)", line)
        if m:
            abbr = _norm_abbr(m.group(1))
            mid = abbr_to_member.get(abbr)
            if mid:
                attended.add(mid)
    return attended, absent


@bp.route("/")
def index():
    engine = _get_engine()
    metrics_by_mal: Dict[int, Any] = {}
    with Session(engine) as session:
        lthing = _selected_lthing(session)
        issues = session.execute(
            select(models.ThingmalalistiMal)
            .where(models.ThingmalalistiMal.ingest_lthing == lthing)
            .order_by(models.ThingmalalistiMal.attr_malsnumer)
        ).scalars().all()
        members = session.execute(
            select(models.ThingmannalistiThingmadur).where(models.ThingmannalistiThingmadur.ingest_lthing == lthing)
        ).scalars().all()
        seats = session.execute(
            select(MemberSeat).where(MemberSeat.lthing == lthing)
        ).scalars().all()
        display_seats = seats
        if not display_seats:
            try:
                fallback_lthing = session.execute(
                    select(func.max(MemberSeat.lthing))
                ).scalar_one_or_none()
                if fallback_lthing and fallback_lthing != lthing:
                    display_seats = session.execute(
                        select(MemberSeat).where(MemberSeat.lthing == fallback_lthing)
                    ).scalars().all()
            except Exception:
                display_seats = seats

        votes = session.execute(
            select(models.AtkvaedagreidslurAtkvaedagreidsla)
            .where(models.AtkvaedagreidslurAtkvaedagreidsla.ingest_lthing == lthing)
        ).scalars().all()

        votes_by_mal: Dict[int, List[Any]] = defaultdict(list)
        for vote in votes:
            if vote.attr_malsnumer is not None:
                votes_by_mal[int(vote.attr_malsnumer)].append(vote)
        for vlist in votes_by_mal.values():
            vlist.sort(key=lambda v: v.attr_atkvaedagreidslunumer or 0)

        # Aggregate vote_details to compute counts for each vote_num
        counts_by_vote: Dict[int, Dict[str, int]] = defaultdict(lambda: {"ja": 0, "nei": 0, "greiddu": 0})
        vote_counts_rows = session.execute(
            select(VoteDetail.vote_num, VoteDetail.vote, func.count(VoteDetail.id))
            .where(VoteDetail.lthing == lthing)
            .group_by(VoteDetail.vote_num, VoteDetail.vote)
        ).all()
        for vote_num, vote_txt, cnt in vote_counts_rows:
            if vote_txt == "já":
                counts_by_vote[vote_num]["ja"] += cnt
            elif vote_txt == "nei":
                counts_by_vote[vote_num]["nei"] += cnt
            else:
                counts_by_vote[vote_num]["greiddu"] += cnt

        # Build lookup of stored þingskjöl (for flutningsmenn enrichment)
        documents = session.execute(
            select(models.ThingskjalalistiThingskjal)
            .where(models.ThingskjalalistiThingskjal.ingest_lthing == lthing)
        ).scalars().all()
        docs_by_nr: Dict[int, Any] = {}
        for doc in documents:
            attach_flutningsmenn(doc)
            if doc.attr_skjalsnumer is not None:
                docs_by_nr[int(doc.attr_skjalsnumer)] = doc

        docs_by_mal: Dict[int, List[Any]] = defaultdict(list)
        first_doc_date: Dict[int, dt.datetime] = {}
        first_answer_date: Dict[int, dt.datetime] = {}
        # Pre-fetched issue documents are stored in IssueDocument; join to flutningsmenn via skjalnr
        issue_docs = session.execute(
            select(IssueDocument).where(IssueDocument.lthing == lthing)
        ).scalars().all()
        if manual_models and hasattr(manual_models, "IssueMetrics"):
            try:
                lthing_val = _selected_lthing(session)
                if lthing_val is not None:
                    for m in session.execute(
                        select(manual_models.IssueMetrics).where(manual_models.IssueMetrics.lthing == lthing_val)
                    ).scalars().all():
                        if m.malnr is not None:
                            metrics_by_mal[int(m.malnr)] = m
            except Exception:
                metrics_by_mal = {}
        for d in issue_docs:
            stored = docs_by_nr.get(d.skjalnr or -1)
            docs_by_mal[d.malnr or -1].append(type("DocProxy", (), {
                "leaf_skjalategund": d.skjalategund or (f"Skjal {d.skjalnr}" if d.skjalnr else "Skjal"),
                "leaf_utbyting": d.utbyting,
                "leaf_athugasemd": getattr(stored, "leaf_athugasemd", None) if stored else None,
                "leaf_slod_html": d.slod_html or (getattr(stored, "leaf_slod_html", None) if stored else None),
                "leaf_slod_pdf": d.slod_pdf or (getattr(stored, "leaf_slod_pdf", None) if stored else None),
                "leaf_slod_xml": d.slod_xml or (getattr(stored, "leaf_slod_xml", None) if stored else None),
                "_flutningsmenn": getattr(stored, "_flutningsmenn", []),
                "attr_skjalsnumer": d.skjalnr,
            }))
            # track first doc date and first answer date
            utb_dt = parse_date(d.utbyting)
            if utb_dt and d.malnr is not None:
                malnr_int = int(d.malnr)
                existing = first_doc_date.get(malnr_int)
                first_doc_date[malnr_int] = utb_dt if existing is None or utb_dt < existing else existing
                if d.skjalategund and "svar" in d.skjalategund.lower():
                    ans_existing = first_answer_date.get(malnr_int)
                    first_answer_date[malnr_int] = utb_dt if ans_existing is None or utb_dt < ans_existing else ans_existing

        for malnr, items in list(docs_by_mal.items()):
            items.sort(key=lambda d: parse_date(getattr(d, "leaf_utbyting", None)) or dt.datetime.min)

    # map member id -> party name for display with speeches
    seat_by_member: Dict[int, MemberSeat] = {}
    for seat in display_seats:
        if seat.member_id is None:
            continue
        try:
            mid = int(seat.member_id)
        except Exception:
            continue
        existing = seat_by_member.get(mid)
        inn_dt = parse_date(seat.inn) or dt.datetime.min
        existing_dt = parse_date(existing.inn) if existing else None
        if existing is None or (existing_dt or dt.datetime.min) < inn_dt:
            seat_by_member[mid] = seat

    party_by_member: Dict[int, str] = {}
    for m in members:
        if m.attr_id is None:
            continue
        try:
            mid = int(m.attr_id)
        except Exception:
            continue
        seat = seat_by_member.get(mid)
        party = None
        if seat and seat.party_name:
            party = seat.party_name
        elif seat and seat.party_id:
            party = str(seat.party_id)
        elif m.leaf_skammstofun:
            party = m.leaf_skammstofun
        if party:
            party_by_member[mid] = party

    cache_dir = _cache_dir()
    speeches_cache: Dict[str, List[Dict[str, Any]]] = {}
    speeches_by_mal: Dict[int, List[Dict[str, Any]]] = {}
    parties_by_issue: Dict[int, set] = defaultdict(set)
    type_by_issue: Dict[int, str] = {}
    answered_by_issue: Dict[int, str] = {}
    minister_map = _minister_map(session, lthing)
    name_to_id = {name: info["id"] for name, info in minister_map.items()}
    for issue in issues:
        key = issue.attr_malsnumer
        if key is None:
            continue
        typ = canonical_issue_type(issue.leaf_malstegund_heiti2 or issue.leaf_malstegund_heiti)
        typ_lower = typ.lower()
        m = metrics_by_mal.get(int(key))
        if m is not None and getattr(m, "answer_latency", None) is not None:
            issue._answer_latency = int(m.answer_latency)
            if getattr(m, "answer_status", None):
                answered_by_issue[int(key)] = str(m.answer_status)
        else:
            latency_days = None
            is_written_question = (
                getattr(issue, "attr_malsflokkur", None) == "A"
                and canonical_issue_type(issue.leaf_malstegund_heiti2).casefold() == WRITTEN_QUESTION_LABEL.casefold()
            )
            answer_dt: Optional[dt.date] = None
            if is_written_question:
                start_dt = first_doc_date.get(int(key))
                docs_for_issue = docs_by_mal.get(int(key), [])
                question_dt: Optional[dt.date] = None
                for doc in docs_for_issue:
                    stype = (getattr(doc, "leaf_skjalategund", "") or "").lower()
                    utb = parse_date(getattr(doc, "leaf_utbyting", None))
                    is_question_doc = ("fsp" in stype) or ("fyrirspurn" in stype)
                    is_answer_doc = ("svar" in stype) and not is_question_doc
                    if is_question_doc and utb:
                        if question_dt is None or utb.date() < question_dt:
                            question_dt = utb.date()
                    if is_answer_doc:
                        attn = getattr(doc, "leaf_athugasemd", None) or ""
                        cand_dt = prefer_athugasemd_date(attn) or utb
                        if cand_dt and (answer_dt is None or cand_dt.date() < answer_dt):
                            answer_dt = cand_dt.date()
                if question_dt is None and start_dt:
                    question_dt = start_dt.date()
                if question_dt:
                    end_date = answer_dt or dt.date.today()
                    latency_days = business_days_between(question_dt, end_date)
                answered_by_issue[int(key)] = "svarað" if answer_dt else "ósvarað"
            issue._answer_latency = latency_days
        speeches_by_mal[int(key)] = _cached_speeches(
            getattr(issue, "leaf_xml", None),
            cache_dir,
            speeches_cache,
            party_by_member=party_by_member,
            name_to_member_id=name_to_id,
        )
        # party attribution via primary flutningsmaður on earliest doc (if available)
        docs_for_issue = docs_by_mal.get(int(key), [])
        for doc in docs_for_issue:
            mid = flutningsmenn_primary_id(doc)
            if mid is not None:
                party_name = party_by_member.get(int(mid))
                if party_name:
                    parties_by_issue[int(key)].add(party_name)
                break
        # store type for counting
        if typ:
            type_by_issue[int(key)] = typ
        # answered_by_issue is set above (metrics or fallback)

    # aggregate counts
    party_counts: Dict[str, int] = defaultdict(int)
    type_counts: Dict[str, int] = defaultdict(int)
    answer_counts: Dict[str, int] = defaultdict(int)
    issue_parties_map: Dict[int, List[str]] = {}
    for malnr, parties in parties_by_issue.items():
        for p in parties:
            party_counts[p] += 1
        issue_parties_map[malnr] = sorted(parties, key=icelandic_sort_key)
    for malnr, typ in type_by_issue.items():
        type_counts[typ] += 1
    for malnr, status in answered_by_issue.items():
        answer_counts[status] += 1

    error = None
    if not issues:
        error = "Engin gögn fundust fyrir þetta löggjafarþing."
    return render_template(
        "index.html",
        issues=issues,
        votes_by_mal=votes_by_mal,
        docs_by_mal=docs_by_mal,
        vote_counts=counts_by_vote,
        current_lthing=lthing,
        speeches_by_mal=speeches_by_mal,
        party_counts=dict(sorted(party_counts.items(), key=lambda kv: icelandic_sort_key(kv[0]))),
        type_counts=dict(sorted(type_counts.items(), key=lambda kv: icelandic_sort_key(kv[0]))),
        issue_parties=issue_parties_map,
        issue_types=type_by_issue,
        issue_answer_status=answered_by_issue,
        answer_counts=answer_counts,
        written_question_label=WRITTEN_QUESTION_LABEL,
        error=error,
    )


@bp.route("/members")
def members():
    engine = _get_engine()
    speech_counts: Dict[int, int] = {}
    with Session(engine) as session:
        lthing = _selected_lthing(session)
        people = session.execute(
            select(models.ThingmannalistiThingmadur)
            .where(models.ThingmannalistiThingmadur.ingest_lthing == lthing)
            .order_by(models.ThingmannalistiThingmadur.leaf_nafn)
        ).scalars().all()
        seats = session.execute(
            select(MemberSeat).where(MemberSeat.lthing == lthing)
        ).scalars().all()
        mal_rows = session.execute(
            select(models.ThingmalalistiMal).where(models.ThingmalalistiMal.ingest_lthing == lthing)
        ).scalars().all()
        issue_docs = session.execute(
            select(IssueDocument).where(IssueDocument.lthing == lthing)
        ).scalars().all()
        docs_by_nr = session.execute(
            select(models.ThingskjalalistiThingskjal).where(models.ThingskjalalistiThingskjal.ingest_lthing == lthing)
        ).scalars().all()
        # vote attendance: prefer materialized vote_session; fallback to raw votes
        vote_sessions = []
        if manual_models and hasattr(manual_models, "VoteSession"):
            vote_sessions = session.execute(
                select(manual_models.VoteSession.vote_num, manual_models.VoteSession.time)
                .where(manual_models.VoteSession.lthing == lthing)
            ).all()
        if not vote_sessions:
            vote_sessions = session.execute(
                select(
                    models.AtkvaedagreidslurAtkvaedagreidsla.attr_atkvaedagreidslunumer,
                    models.AtkvaedagreidslurAtkvaedagreidsla.leaf_timi,
                ).where(models.AtkvaedagreidslurAtkvaedagreidsla.ingest_lthing == lthing)
            ).all()
        all_vote_details = session.execute(
            select(VoteDetail.vote_num, VoteDetail.voter_id, VoteDetail.vote)
            .where(VoteDetail.lthing == lthing)
        ).all()
        vote_nums_with_votes = {vd[0] for vd in all_vote_details if vd[0] is not None}
        vote_sessions = [vs for vs in vote_sessions if vs[0] in vote_nums_with_votes]
        nefnd_members_rows = session.execute(
            select(NefndMember).where(NefndMember.lthing == lthing)
        ).scalars().all()
        committee_meetings = []
        committee_attendance_rows = []
        if manual_models and hasattr(manual_models, "CommitteeMeeting"):
            try:
                committee_meetings = session.execute(
                    select(manual_models.CommitteeMeeting).where(manual_models.CommitteeMeeting.lthing == lthing)
                ).scalars().all()
            except OperationalError:
                committee_meetings = []
        if manual_models and hasattr(manual_models, "CommitteeAttendance"):
            try:
                committee_attendance_rows = session.execute(
                    select(manual_models.CommitteeAttendance).where(manual_models.CommitteeAttendance.lthing == lthing)
                ).scalars().all()
            except OperationalError:
                committee_attendance_rows = []
        if manual_models and hasattr(manual_models, "Speech"):
            try:
                rows = session.execute(
                    select(manual_models.Speech.member_id, func.count())
                    .where(manual_models.Speech.lthing == lthing, manual_models.Speech.member_id.is_not(None))
                    .group_by(manual_models.Speech.member_id)
                ).all()
                speech_counts = {int(mid): cnt for mid, cnt in rows if mid is not None}
            except Exception:
                speech_counts = {}
        minister_map = _minister_map(session, lthing)

    docs_map = {}
    for d in docs_by_nr:
        attach_flutningsmenn(d)
        if d.attr_skjalsnumer is not None:
            docs_map[int(d.attr_skjalsnumer)] = d

    # Build abbrev map for attendance parsing
    abbr_to_member: Dict[str, int] = {}
    for p in people:
        if p.attr_id is not None and p.leaf_skammstofun:
            abbr_to_member[_norm_abbr(p.leaf_skammstofun)] = int(p.attr_id)

    # intervals per member for attendance filtering
    intervals_by_member = _effective_intervals(seats)

    # Committee attendance: compute totals using committee membership intervals and meeting schedule
    attendance_attended: Dict[int, int] = defaultdict(int)
    attendance_total: Dict[int, int] = defaultdict(int)
    committee_intervals: Dict[int, Dict[int, List[Tuple[Optional[dt.date], Optional[dt.date]]]]] = defaultdict(lambda: defaultdict(list))
    official_members: Dict[int, set] = defaultdict(set)

    def role_rank(role: Optional[str]) -> int:
        if not role:
            return 99
        lr = role.lower().replace(".", "").strip()
        if "formaður" in lr and "vara" not in lr:
            return 0
        if lr.startswith("1 varaformaður") or "1. varaformaður" in lr or "1 varaformadur" in lr:
            return 1
        if lr.startswith("2 varaformaður") or "2. varaformaður" in lr or "2 varaformadur" in lr:
            return 2
        if "nefndarmaður" in lr:
            return 3
        if "áheyrnarfulltrú" in lr:
            return 4
        if "varamaður" in lr:
            return 5
        return 99

    for nm in nefnd_members_rows:
        if nm.nefnd_id is None or nm.member_id is None:
            continue
        rank = role_rank(nm.role)
        if rank > 3:
            continue  # only official members
        nid = int(nm.nefnd_id)
        mid = int(nm.member_id)
        inn_dt = parse_date(nm.inn)
        ut_dt = parse_date(nm.ut)
        committee_intervals[nid][mid].append((inn_dt.date() if inn_dt else None, ut_dt.date() if ut_dt else None))
        official_members[nid].add(mid)

    present_map: Dict[int, set] = defaultdict(set)
    for ca in committee_attendance_rows:
        if ca.meeting_id is None or ca.member_id is None:
            continue
        if ca.status in ("present", "proxy_present"):
            present_map[int(ca.meeting_id)].add(int(ca.member_id))
    meetings_info: List[Tuple[int, Optional[int], Optional[dt.datetime]]] = []
    for mt in committee_meetings:
        dt_val = _parse_iso(mt.start_time) or parse_date(mt.start_time)
        meetings_info.append((mt.id, mt.nefnd_id, dt_val))

    def in_date_interval(d: dt.date, intervals: List[Tuple[Optional[dt.date], Optional[dt.date]]]) -> bool:
        if not intervals:
            return True
        for inn, ut in intervals:
            if inn and d < inn:
                continue
            if ut and d > ut:
                continue
            return True
        return False

    for meeting_id, nefnd_id, dt_val in meetings_info:
        if nefnd_id is None or dt_val is None:
            continue
        d = dt_val.date()
        members = official_members.get(int(nefnd_id), set())
        for mid in members:
            if not in_date_interval(d, committee_intervals.get(int(nefnd_id), {}).get(mid, [])):
                continue
            intervals = intervals_by_member.get(mid, [])
            if intervals and not _in_intervals(dt_val, intervals):
                continue
            attendance_total[mid] += 1
            if mid in present_map.get(meeting_id, set()):
                attendance_attended[mid] += 1

    seat_by_member: Dict[int, MemberSeat] = {}
    for seat in seats:
        key = int(seat.member_id)
        existing = seat_by_member.get(key)
        inn_dt = parse_date(seat.inn) or dt.datetime.min
        existing_dt = parse_date(existing.inn) if existing else None
        if existing is None or (existing_dt or dt.datetime.min) < inn_dt:
            seat_by_member[key] = seat

    # intervals per member for attendance filtering
    intervals_by_member = _effective_intervals(seats)

    for p in people:
        p.current_seat = seat_by_member.get(int(p.attr_id)) if p.attr_id is not None else None
        p.cv_slug = quote((p.leaf_nafn or "").replace(" ", "_"))
        p.issues = []

    mal_by_key = {(m.attr_malsnumer, getattr(m, "attr_malsflokkur", None)): m for m in mal_rows if m.attr_malsnumer is not None}
    docs_by_mal: Dict[tuple, List[IssueDocument]] = defaultdict(list)
    for d in issue_docs:
        if d.malnr is not None:
            key = (int(d.malnr), getattr(d, "malflokkur", None))
            docs_by_mal[key].append(d)

    # Determine primary flutningsmaður per mál (choose earliest doc / lowest skjalnr, skip answers)
    mal_primary: Dict[tuple, Dict[str, Any]] = {}
    for d in issue_docs:
        skjalnr = d.skjalnr
        if skjalnr is None:
            continue
        # Prefer non-answer docs for primary attribution but allow fallback
        if d.skjalategund and "svar" in d.skjalategund.lower():
            continue
        th_doc = docs_map.get(skjalnr)
        if not th_doc:
            continue
        mid = flutningsmenn_primary_id(th_doc)
        if mid is None:
            continue
        mal_key = (int(d.malnr), getattr(d, "malflokkur", None))
        existing = mal_primary.get(mal_key)
        if existing is None or (existing.get("skjalnr") or 1e9) > skjalnr:
            mal_primary[mal_key] = {"member_id": mid, "issue_doc": d, "skjalnr": skjalnr}

    # If some mál still missing primary (no non-answer docs), allow svar docs
    for d in issue_docs:
        skjalnr = d.skjalnr
        if skjalnr is None:
            continue
        if not (d.skjalategund and "svar" in d.skjalategund.lower()):
            continue
        mal_key = (int(d.malnr), getattr(d, "malflokkur", None))
        if mal_key in mal_primary:
            continue
        th_doc = docs_map.get(skjalnr)
        if not th_doc:
            continue
        mid = flutningsmenn_primary_id(th_doc)
        if mid is None:
            continue
        mal_primary[mal_key] = {"member_id": mid, "issue_doc": d, "skjalnr": skjalnr}

    # Build mapping member_id -> set of malnr where member is primary flutningsmaður
    member_issues: Dict[int, Dict[tuple, Dict[str, Any]]] = defaultdict(dict)
    for mal_key, info in mal_primary.items():
        mid = info["member_id"]
        member_issues[mid][mal_key] = {"issue_doc": info["issue_doc"]}

    # Attach issue list to members
    for p in people:
        if p.attr_id is None:
            continue
        mid = int(p.attr_id)
        intervals = intervals_by_member.get(mid, [])
        has_intervals = bool(intervals)
        issues_for_member = member_issues.get(int(p.attr_id), {})
        items = []
        for mal_key, data in issues_for_member.items():
            mal = mal_by_key.get(mal_key)
            if not mal:
                continue
            # determine answer link if any doc of type svar
            docs = docs_by_mal.get(mal_key, [])
            answer_link = None
            for d in docs:
                if d.skjalategund and "svar" in d.skjalategund.lower():
                    answer_link = d.slod_html or d.slod_xml
                    break
            items.append({
                "malnr": mal_key[0],
                "title": mal.leaf_malsheiti,
                "html": mal.leaf_html,
                "xml": mal.leaf_xml,
                "answer": answer_link,
            })
        # sort by malnr
        items.sort(key=lambda x: x["malnr"])
        p.issues = items
        # vote attendance within seat intervals; exclude notified absences
        if not vote_sessions:
            vote_nums_in_interval = set()
        elif not has_intervals:
            vote_nums_in_interval = {vm[0] for vm in vote_sessions}
        else:
            vote_nums_in_interval = {
                vn for vn, ts in (
                    (vm[0], parse_date(vm[1]) or _parse_iso(vm[1]))
                    for vm in vote_sessions
                )
                if _in_intervals(ts, intervals)
            }
        member_votes = [
            vd for vd in all_vote_details
            if vd[1] == mid and vd[0] in vote_nums_in_interval
        ]
        vc = sum(
            1 for vd in member_votes
            if vd[2] in ("já", "nei", "greiðir ekki atkvæði") and vd[2] != "boðaði fjarvist"
        )
        total_votes_for_member = len(vote_nums_in_interval) if vote_nums_in_interval else len({vs[0] for vs in vote_sessions})
        p.vote_att_count = vc
        p.vote_att_total = total_votes_for_member
        p.vote_att_pct = (vc / total_votes_for_member * 100) if total_votes_for_member else None
        # committee attendance within seat intervals
        ca_total = attendance_total.get(mid, 0)
        ca_att = attendance_attended.get(mid, 0)
        p.committee_att_count = ca_att
        p.committee_att_total = ca_total
        p.committee_att_pct = (ca_att / ca_total * 100) if ca_total else None
        p.speech_count = speech_counts.get(mid, 0)

    if minister_map:
        for name, info in minister_map.items():
            pid = info.get("id")
            if pid is None:
                continue
            seat_proxy = type("SeatProxy", (), {
                "party_name": "Ráðherrar",
                "type": (info.get("role") or "ráðherra"),
                "kjordaemi_name": None,
                "inn": None,
                "ut": None,
            })
            pseudo = type("MemberProxy", (), {
                "attr_id": pid,
                "leaf_nafn": name,
                "leaf_skammstofun": None,
                "leaf_faedingardagur": None,
                "current_seat": seat_proxy,
                "vote_att_pct": None,
                "committee_att_pct": None,
                "speech_count": info.get("count", 0),
                "issues": [],
                "cv_slug": None,
                "is_pseudo": True,
            })
            people.append(pseudo)

    if manual_models and hasattr(manual_models, "Speech"):
        try:
            role_lower = func.lower(manual_models.Speech.speaker_role)
            missing_rows = session.execute(
                select(
                    manual_models.Speech.member_id,
                    func.max(manual_models.Speech.speaker_name),
                    func.max(manual_models.Speech.speaker_role),
                ).where(
                    manual_models.Speech.lthing == lthing,
                    manual_models.Speech.member_id.is_not(None),
                    manual_models.Speech.speaker_name.is_not(None),
                    manual_models.Speech.speaker_role.is_not(None),
                    or_(role_lower.like("%ráðherra%"), role_lower.like("%radherra%")),
                ).group_by(manual_models.Speech.member_id)
            ).all()
            member_ids = {int(p.attr_id) for p in people if getattr(p, "attr_id", None) is not None}
            for mid, name, role in missing_rows:
                if mid is None or int(mid) in member_ids or not name:
                    continue
                seat_proxy = type("SeatProxy", (), {
                    "party_name": "Ráðherrar",
                    "type": role or "ráðherra",
                    "kjordaemi_name": None,
                    "inn": None,
                    "ut": None,
                })
                pseudo = type("MemberProxy", (), {
                    "attr_id": int(mid),
                    "leaf_nafn": name,
                    "leaf_skammstofun": None,
                    "leaf_faedingardagur": None,
                    "current_seat": seat_proxy,
                    "vote_att_pct": None,
                    "committee_att_pct": None,
                    "speech_count": speech_counts.get(int(mid), 0),
                    "issues": [],
                    "cv_slug": None,
                    "is_pseudo": True,
                })
                people.append(pseudo)
        except Exception:
            pass

    def party_for_member(m):
        if getattr(m, "current_seat", None) and m.current_seat.party_name:
            return m.current_seat.party_name
        return "Óflokkaðir"

    parties: Dict[str, List[Any]] = defaultdict(list)
    for p in people:
        parties[party_for_member(p)].append(p)
    # Sort party keys with Icelandic order
    parties = dict(sorted(parties.items(), key=lambda kv: icelandic_sort_key(kv[0])))

    for grp in parties.values():
        grp.sort(key=lambda x: (
            1 if getattr(x, "current_seat", None) and getattr(x.current_seat, "type", "") == "varamaður" else 0,
            icelandic_sort_key(x.leaf_nafn or ""),
        ))

    error = None
    if not people:
        error = "Engir þingmenn fundust fyrir þetta löggjafarþing."
    return render_template("members.html", parties=parties, current_lthing=lthing, error=error)


@bp.route("/member/<int:member_id>")
def member_detail(member_id: int):
    engine = _get_engine()
    Speech = getattr(manual_models, "Speech", None) if manual_models else None
    MemberSeat = getattr(manual_models, "MemberSeat", None) if manual_models else None
    NefndMember = getattr(manual_models, "NefndMember", None) if manual_models else None

    with Session(engine) as session:
        lthing = _selected_lthing(session)
        if member_id < 0:
            minister_map = _minister_map(session, lthing)
            minister_by_id = {info["id"]: {"name": name, **info} for name, info in minister_map.items()}
            info = minister_by_id.get(member_id)
            if not info:
                return render_template(
                    "member.html",
                    current_lthing=lthing,
                    member=None,
                    seat=None,
                    committees=[],
                    summary=None,
                    issues=[],
                    speeches=[],
                    error="Þingmaður fannst ekki.",
                )
            member = type("MemberProxy", (), {
                "attr_id": member_id,
                "leaf_nafn": info["name"],
                "leaf_skammstofun": None,
                "is_pseudo": True,
                "cv_slug": None,
            })
            seat = type("SeatProxy", (), {
                "party_name": "Ráðherrar",
                "type": info.get("role") or "ráðherra",
                "kjordaemi_name": None,
                "inn": None,
                "ut": None,
            })
            committees = []
            summary = None
            speeches = []
            issues = []
            if Speech:
                filters = [Speech.member_id.is_(None), Speech.speaker_name == info["name"]]
                if lthing is not None:
                    filters.append(Speech.lthing == lthing)
                total_speeches = session.execute(
                    select(func.count()).where(*filters)
                ).scalar() or 0
                total_words = session.execute(
                    select(func.sum(Speech.word_count)).where(*(filters + [Speech.word_count.is_not(None)]))
                ).scalar() or 0
                avg_wpm = session.execute(
                    select(func.avg(Speech.words_per_minute)).where(*(filters + [Speech.words_per_minute.is_not(None)]))
                ).scalar()
                last_speech = session.execute(
                    select(func.max(Speech.start_time)).where(*filters)
                ).scalar()
                summary = {
                    "total": total_speeches,
                    "total_words": total_words,
                    "avg_words": (total_words / total_speeches) if total_speeches else None,
                    "avg_wpm": avg_wpm,
                    "last_speech": last_speech,
                }
                speech_rows = session.execute(
                    select(Speech).where(*filters).order_by(Speech.start_time.desc()).limit(10)
                ).scalars().all()
                for s in speech_rows:
                    title = getattr(s, "issue_malsheiti", None) or getattr(s, "kind", None) or getattr(s, "umraeda", None)
                    speeches.append({
                        "id": s.id,
                        "title": title,
                        "date": getattr(s, "date", None),
                        "words": getattr(s, "word_count", None),
                        "wpm": getattr(s, "words_per_minute", None),
                        "duration": _fmt_seconds(getattr(s, "duration_seconds", None)),
                        "html": getattr(s, "html_url", None),
                        "xml": getattr(s, "xml_url", None),
                    })
            return render_template(
                "member.html",
                current_lthing=lthing,
                member=member,
                seat=seat,
                committees=committees,
                summary=summary,
                issues=issues,
                speeches=speeches,
                error=None,
            )
        member = session.execute(
            select(models.ThingmannalistiThingmadur).where(
                models.ThingmannalistiThingmadur.attr_id == member_id,
                models.ThingmannalistiThingmadur.ingest_lthing == lthing,
            )
        ).scalar_one_or_none()
        is_pseudo = False
        seat = None
        committees = []
        issues = []
        if member is None:
            role_lower = func.lower(Speech.speaker_role) if Speech else None
            info = None
            if Speech and role_lower is not None:
                info = session.execute(
                    select(
                        Speech.speaker_name,
                        func.max(Speech.speaker_role),
                    ).where(
                        Speech.member_id == member_id,
                        Speech.speaker_name.is_not(None),
                        Speech.speaker_role.is_not(None),
                        or_(role_lower.like("%ráðherra%"), role_lower.like("%radherra%")),
                    )
                ).first()
            if info and info[0]:
                member = type("MemberProxy", (), {
                    "attr_id": member_id,
                    "leaf_nafn": info[0],
                    "leaf_skammstofun": None,
                    "is_pseudo": True,
                    "cv_slug": None,
                })
                seat = type("SeatProxy", (), {
                    "party_name": "Ráðherrar",
                    "type": info[1] or "ráðherra",
                    "kjordaemi_name": None,
                    "inn": None,
                    "ut": None,
                })
                is_pseudo = True
            else:
                return render_template(
                    "member.html",
                    current_lthing=lthing,
                    member=None,
                    seat=None,
                    committees=[],
                    summary=None,
                    issues=[],
                    speeches=[],
                    error="Þingmaður fannst ekki.",
                )

        if not is_pseudo and MemberSeat:
            seats = session.execute(
                select(MemberSeat).where(MemberSeat.lthing == lthing, MemberSeat.member_id == member_id)
            ).scalars().all()
            if seats:
                seat = max(seats, key=lambda s: parse_date(s.inn) or dt.datetime.min)

        if not is_pseudo and NefndMember:
            nm_rows = session.execute(
                select(NefndMember).where(NefndMember.lthing == lthing, NefndMember.member_id == member_id)
            ).scalars().all()
            if nm_rows:
                nefnd_ids = [int(nm.nefnd_id) for nm in nm_rows if nm.nefnd_id is not None]
                nefnd_map = {}
                if nefnd_ids:
                    nefnd_rows = session.execute(
                        select(models.NefndirNefnd).where(models.NefndirNefnd.attr_id.in_(nefnd_ids))
                    ).scalars().all()
                    nefnd_map = {int(n.attr_id): n for n in nefnd_rows if n.attr_id is not None}
                for nm in nm_rows:
                    nid = int(nm.nefnd_id) if nm.nefnd_id is not None else None
                    nefnd = nefnd_map.get(nid) if nid is not None else None
                    committees.append({
                        "id": nid,
                        "name": getattr(nefnd, "leaf_heiti", None) or nm.name or "Ónefnd",
                        "role": nm.role,
                    })
                committees.sort(key=lambda c: icelandic_sort_key(c["name"] or ""))

        summary = None
        speeches = []
        if Speech:
            filters = [Speech.member_id == member_id]
            if lthing is not None:
                filters.append(Speech.lthing == lthing)
            total_speeches = session.execute(
                select(func.count()).where(*filters)
            ).scalar() or 0
            total_words = session.execute(
                select(func.sum(Speech.word_count)).where(*(filters + [Speech.word_count.is_not(None)]))
            ).scalar() or 0
            avg_wpm = session.execute(
                select(func.avg(Speech.words_per_minute)).where(*(filters + [Speech.words_per_minute.is_not(None)]))
            ).scalar()
            last_speech = session.execute(
                select(func.max(Speech.start_time)).where(*filters)
            ).scalar()
            summary = {
                "total": total_speeches,
                "total_words": total_words,
                "avg_words": (total_words / total_speeches) if total_speeches else None,
                "avg_wpm": avg_wpm,
                "last_speech": last_speech,
            }
            speech_rows = session.execute(
                select(Speech).where(*filters).order_by(Speech.start_time.desc()).limit(10)
            ).scalars().all()
            for s in speech_rows:
                title = getattr(s, "issue_malsheiti", None) or getattr(s, "kind", None) or getattr(s, "umraeda", None)
                speeches.append({
                    "id": s.id,
                    "title": title,
                    "date": getattr(s, "date", None),
                    "words": getattr(s, "word_count", None),
                    "wpm": getattr(s, "words_per_minute", None),
                    "duration": _fmt_seconds(getattr(s, "duration_seconds", None)),
                    "html": getattr(s, "html_url", None),
                    "xml": getattr(s, "xml_url", None),
                })

        include_issues = manual_models and hasattr(manual_models, "IssueDocument")
        if include_issues and (not is_pseudo or member_id > 0):
            docs = session.execute(
                select(manual_models.IssueDocument).where(manual_models.IssueDocument.lthing == lthing)
            ).scalars().all()
            mal_rows = session.execute(
                select(models.ThingmalalistiMal).where(models.ThingmalalistiMal.ingest_lthing == lthing)
            ).scalars().all()
            documents = session.execute(
                select(models.ThingskjalalistiThingskjal)
                .where(models.ThingskjalalistiThingskjal.ingest_lthing == lthing)
            ).scalars().all()
            docs_by_nr: Dict[int, Any] = {}
            for doc in documents:
                attach_flutningsmenn(doc)
                if doc.attr_skjalsnumer is not None:
                    docs_by_nr[int(doc.attr_skjalsnumer)] = doc
            mal_by_key = {(m.attr_malsnumer, getattr(m, "attr_malsflokkur", None)): m for m in mal_rows if m.attr_malsnumer is not None}
            docs_by_mal: Dict[tuple, List[Any]] = defaultdict(list)
            for d in docs:
                if d.malnr is not None:
                    key = (int(d.malnr), getattr(d, "malflokkur", None))
                    docs_by_mal[key].append(d)

            # Determine primary flutningsmaður per mál (prefer non-answer docs).
            mal_primary: Dict[tuple, Dict[str, Any]] = {}
            for d in docs:
                skjalnr = d.skjalnr
                if skjalnr is None:
                    continue
                if d.skjalategund and "svar" in d.skjalategund.lower():
                    continue
                th_doc = docs_by_nr.get(skjalnr)
                if not th_doc:
                    continue
                mid = flutningsmenn_primary_id(th_doc)
                if mid is None:
                    continue
                mal_key = (int(d.malnr), getattr(d, "malflokkur", None))
                existing = mal_primary.get(mal_key)
                if existing is None or (existing.get("skjalnr") or 1e9) > skjalnr:
                    mal_primary[mal_key] = {"member_id": mid, "issue_doc": d, "skjalnr": skjalnr}

            for d in docs:
                skjalnr = d.skjalnr
                if skjalnr is None:
                    continue
                if not (d.skjalategund and "svar" in d.skjalategund.lower()):
                    continue
                mal_key = (int(d.malnr), getattr(d, "malflokkur", None))
                if mal_key in mal_primary:
                    continue
                th_doc = docs_by_nr.get(skjalnr)
                if not th_doc:
                    continue
                mid = flutningsmenn_primary_id(th_doc)
                if mid is None:
                    continue
                mal_primary[mal_key] = {"member_id": mid, "issue_doc": d, "skjalnr": skjalnr}

            for mal_key, info in mal_primary.items():
                if int(info["member_id"]) != member_id:
                    continue
                mal = mal_by_key.get(mal_key)
                if not mal:
                    continue
                docs_for_mal = docs_by_mal.get(mal_key, [])
                answer_link = None
                for d in docs_for_mal:
                    if d.skjalategund and "svar" in d.skjalategund.lower():
                        answer_link = d.slod_html or d.slod_xml
                        break
                issues.append({
                    "malnr": mal_key[0],
                    "title": mal.leaf_malsheiti,
                    "html": mal.leaf_html,
                    "xml": mal.leaf_xml,
                    "answer": answer_link,
                })
            issues.sort(key=lambda x: x["malnr"])

        if not is_pseudo:
            member.cv_slug = quote((member.leaf_nafn or "").replace(" ", "_"))
            member.is_pseudo = False

    return render_template(
        "member.html",
        current_lthing=lthing,
        member=member,
        seat=seat,
        committees=committees,
        summary=summary,
        issues=issues,
        speeches=speeches,
        error=None,
    )


def _fundargerd_link(raw_xml: str) -> Optional[str]:
    try:
        root = ET.fromstring(raw_xml)
        # Prefer fundargerð html link
        for node in root.iter():
            if _strip_tag(node.tag) == "fundargerð":
                html = None
                xml = None
                for c in list(node):
                    if _strip_tag(c.tag) == "html" and (c.text or "").strip():
                        html = c.text.strip()
                    if _strip_tag(c.tag) == "xml" and (c.text or "").strip():
                        xml = c.text.strip()
                if html:
                    return html
                if xml:
                    return xml
        # If no explicit fundargerð link, do not fall back to dagskrá html
    except Exception:
        return None
    return None


@bp.route("/members/<int:member_id>/attendance")
def member_attendance(member_id: int):
    engine = _get_engine()
    with Session(engine) as session:
        lthing = _selected_lthing(session)
        member = session.execute(
            select(models.ThingmannalistiThingmadur).where(
                models.ThingmannalistiThingmadur.attr_id == member_id,
                models.ThingmannalistiThingmadur.ingest_lthing == lthing,
            )
        ).scalar_one_or_none()
        if member is None:
            return render_template(
                "member_attendance.html",
                member=None,
                current_lthing=lthing,
                attended=0,
                total=0,
                pct=None,
                records=[],
                vote_expected=[],
                vote_attended=0,
                vote_notified=0,
                vote_absent=0,
                vote_total=0,
                vote_pct=None,
                error="Mætingarskýrsla fannst ekki fyrir þennan þingmann.",
            )
        seats = session.execute(
            select(MemberSeat).where(MemberSeat.lthing == lthing, MemberSeat.member_id == member_id)
        ).scalars().all()
        nefnd_members = session.execute(
            select(NefndMember).where(NefndMember.lthing == lthing, NefndMember.member_id == member_id)
        ).scalars().all()
        nefndir = session.execute(
            select(models.NefndirNefnd).where(models.NefndirNefnd.ingest_lthing == lthing)
        ).scalars().all()
        meetings = []
        attendance_rows = []
        if manual_models and hasattr(manual_models, "CommitteeMeeting"):
            try:
                meetings = session.execute(
                    select(manual_models.CommitteeMeeting).where(manual_models.CommitteeMeeting.lthing == lthing)
                ).scalars().all()
            except OperationalError:
                meetings = []
        if manual_models and hasattr(manual_models, "CommitteeAttendance"):
            try:
                attendance_rows = session.execute(
                    select(manual_models.CommitteeAttendance).where(
                        manual_models.CommitteeAttendance.lthing == lthing,
                        manual_models.CommitteeAttendance.member_id == member_id,
                    )
                ).scalars().all()
            except OperationalError:
                attendance_rows = []
        people = session.execute(
            select(models.ThingmannalistiThingmadur).where(models.ThingmannalistiThingmadur.ingest_lthing == lthing)
        ).scalars().all()
        vote_sessions = []
        if manual_models and hasattr(manual_models, "VoteSession"):
            vote_sessions = session.execute(
                select(manual_models.VoteSession.vote_num, manual_models.VoteSession.time)
                .where(manual_models.VoteSession.lthing == lthing)
            ).all()
        if not vote_sessions:
            vote_sessions = session.execute(
                select(
                    models.AtkvaedagreidslurAtkvaedagreidsla.attr_atkvaedagreidslunumer,
                    models.AtkvaedagreidslurAtkvaedagreidsla.leaf_timi,
                ).where(models.AtkvaedagreidslurAtkvaedagreidsla.ingest_lthing == lthing)
            ).all()
        all_vote_details = session.execute(
            select(VoteDetail.vote_num, VoteDetail.voter_id, VoteDetail.vote)
            .where(VoteDetail.lthing == lthing)
        ).all() if manual_models else []
        votes_full = session.execute(
            select(models.AtkvaedagreidslurAtkvaedagreidsla)
            .where(models.AtkvaedagreidslurAtkvaedagreidsla.ingest_lthing == lthing)
        ).scalars().all()

    if member is None:
        return "Member not found", 404

    intervals_by_member = _effective_intervals(seats)
    member_intervals = intervals_by_member.get(member_id, [])

    def role_rank(role: Optional[str]) -> int:
        if not role:
            return 99
        lr = role.lower().replace(".", "").strip()
        if "formaður" in lr and "vara" not in lr:
            return 0
        if lr.startswith("1 varaformaður") or "1. varaformaður" in lr or "1 varaformadur" in lr:
            return 1
        if lr.startswith("2 varaformaður") or "2. varaformaður" in lr or "2 varaformadur" in lr:
            return 2
        if "nefndarmaður" in lr:
            return 3
        if "áheyrnarfulltrú" in lr:
            return 4
        if "varamaður" in lr:
            return 5
        return 99

    committee_intervals: Dict[int, List[Tuple[Optional[dt.date], Optional[dt.date]]]] = defaultdict(list)
    for nm in nefnd_members:
        rrank = role_rank(nm.role)
        if rrank > 3:
            continue
        inn_dt = parse_date(nm.inn)
        ut_dt = parse_date(nm.ut)
        committee_intervals[int(nm.nefnd_id)].append((inn_dt.date() if inn_dt else None, ut_dt.date() if ut_dt else None))

    present_map = {ar.meeting_id for ar in attendance_rows if ar.status in ("present", "proxy_present")}
    arrival_map = {ar.meeting_id: ar.arrival_time for ar in attendance_rows if getattr(ar, "arrival_time", None)}
    nefnd_name_map = {int(n.attr_id): (n.leaf_heiti or f"Nefnd {n.attr_id}") for n in nefndir if n.attr_id is not None}
    abbr_map = {_norm_abbr(p.leaf_skammstofun): int(p.attr_id) for p in people if p.attr_id is not None and p.leaf_skammstofun}

    def _plain_text_from_fundargerd(texti: str) -> str:
        txt = re.sub(r"<br\\s*/?>", "\n", texti, flags=re.IGNORECASE)
        txt = re.sub(r"<[^>]+>", "", txt)
        txt = txt.replace("&nbsp;", " ")
        return txt

    def _extract_absence_reason(mt: Any) -> Optional[str]:
        try:
            root = ET.fromstring(mt.raw_xml)
        except Exception:
            return None
        texti = ""
        for t in root.iter():
            if _strip_tag(t.tag) == "texti":
                candidate = t.text or ""
                if len(candidate) > len(texti):
                    texti = candidate
        if not texti:
            return None

        plain = _plain_text_from_fundargerd(texti)
        member_name = (member.leaf_nafn or "").strip()
        if not member_name:
            return None
        member_name_ascii = unicodedata.normalize("NFKD", member_name.casefold())
        member_name_ascii = "".join(ch for ch in member_name_ascii if not unicodedata.combining(ch))

        def _as_ascii(s: str) -> str:
            s2 = unicodedata.normalize("NFKD", s.casefold())
            return "".join(ch for ch in s2 if not unicodedata.combining(ch))

        # Often multiple members are listed on a single line with multiple sentences.
        # Split into clauses and only use the clause containing this member name.
        for raw_line in plain.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for clause in re.split(r"[.;]\s+", line):
                clause = clause.strip()
                if not clause:
                    continue
                ascii_clause = _as_ascii(clause)
                if member_name_ascii not in ascii_clause:
                    continue
                # Return a concise label for the UI.
                if "forfoll" in ascii_clause:  # forföll/forfoll
                    return "forföll"
                if "fjarverandi" in ascii_clause:
                    return "fjarverandi"
                if "fjarvist" in ascii_clause:
                    return "fjarvist"
        return None

    def _extract_member_timings(mt: Any) -> Tuple[Optional[dt.datetime], Optional[str]]:
        try:
            root = ET.fromstring(mt.raw_xml)
        except Exception:
            return None, None
        texti = ""
        for t in root.iter():
            if _strip_tag(t.tag) == "texti":
                candidate = t.text or ""
                if len(candidate) > len(texti):
                    texti = candidate
        if not texti:
            return None, None
        abbr = _norm_abbr(member.leaf_skammstofun) if member and member.leaf_skammstofun else None
        date_part = (_parse_iso(mt.start_time) or parse_date(mt.start_time) or dt.datetime.now()).date()
        arrival_dt = None
        leave_note = None
        lines = re.split(r"<br\s*/?>", texti, flags=re.IGNORECASE)
        name_lower = (member.leaf_nafn or "").lower()
        name_norm = unicodedata.normalize("NFKD", name_lower)
        name_norm = "".join(ch for ch in name_norm if not unicodedata.combining(ch))
        abbr_norm = abbr_ascii = None
        if member and member.leaf_skammstofun:
            abbr_ascii = _norm_abbr(member.leaf_skammstofun)
        for line in lines:
            l = line.lower()
            l_ascii = unicodedata.normalize("NFKD", l)
            l_ascii = "".join(ch for ch in l_ascii if not unicodedata.combining(ch))
            # arrival time after abbr or name
            name_hit = False
            if name_norm:
                first = name_norm.split()[0]
                if first and first in l_ascii:
                    name_hit = True
            abbr_hit = False
            if abbr_ascii:
                if f"({abbr_ascii})" in l_ascii or abbr_ascii in l_ascii:
                    abbr_hit = True
            if name_hit or abbr_hit:
                m = re.search(r"kl\.?\s*(\d{2}:\d{2})", l_ascii, flags=re.IGNORECASE)
                if not m:
                    m = re.search(r"(\d{2}:\d{2})", l_ascii)
                if m and not arrival_dt:
                    try:
                        t = dt.datetime.strptime(m.group(1), "%H:%M").time()
                        arrival_dt = dt.datetime.combine(date_part, t)
                    except Exception:
                        pass
            # departure note
            if name_norm and name_norm.split()[0] in l_ascii and "vék af fundi" in l:
                m2 = re.search(r"(\d{2}:\d{2})\s*-\s*(\d{2}:\d{2})", line)
                if m2:
                    leave_note = f"Vék af fundi {m2.group(1)}–{m2.group(2)}"
        # Fallback: scan whole text for abbr/name then time
        if not arrival_dt:
            full_ascii = unicodedata.normalize("NFKD", texti.lower())
            full_ascii = "".join(ch for ch in full_ascii if not unicodedata.combining(ch))
            targets = []
            if abbr_ascii:
                targets.append(abbr_ascii)
            if name_norm:
                parts = name_norm.split()
                if parts:
                    targets.append(parts[0])
            for tgt in targets:
                idx = full_ascii.find(tgt)
                if idx == -1:
                    continue
                segment = full_ascii[idx: idx + 120]  # look a bit ahead
                m = re.search(r"kl\.?\s*(\d{2}:\d{2})", segment)
                if not m:
                    m = re.search(r"(\d{2}:\d{2})", segment)
                if m:
                    try:
                        t = dt.datetime.strptime(m.group(1), "%H:%M").time()
                        arrival_dt = dt.datetime.combine(date_part, t)
                        break
                    except Exception:
                        pass
        return arrival_dt, leave_note

    def in_date_interval(d: dt.date, intervals: List[Tuple[Optional[dt.date], Optional[dt.date]]]) -> bool:
        if not intervals:
            return True
        for inn, ut in intervals:
            if inn and d < inn:
                continue
            if ut and d > ut:
                continue
            return True
        return False

    records = []
    attended = 0
    total = 0
    for mt in meetings:
        if mt.nefnd_id not in committee_intervals:
            continue
        dt_val = _parse_iso(mt.start_time) or parse_date(mt.start_time)
        if not dt_val:
            continue
        d = dt_val.date()
        if not in_date_interval(d, committee_intervals[int(mt.nefnd_id)]):
            continue
        if member_intervals and not _in_intervals(dt_val, member_intervals):
            continue
        status = "attended" if mt.id in present_map else "missed"
        arrival_dt = None
        leave_note = None
        absence_reason = None
        if mt.id in arrival_map and arrival_map[mt.id]:
            try:
                t = dt.datetime.strptime(arrival_map[mt.id], "%H:%M").time()
                arrival_dt = dt.datetime.combine(d, t)
            except Exception:
                arrival_dt = None
        if arrival_dt is None:
            arrival_dt, leave_note = _extract_member_timings(mt)
        if status == "missed":
            absence_reason = _extract_absence_reason(mt)
        meeting_start = _parse_iso(mt.start_time) or parse_date(mt.start_time)
        if status == "attended" and arrival_dt is None and meeting_start:
            arrival_dt = meeting_start  # fallback to start time if not parsed
        is_late = bool(arrival_dt and meeting_start and arrival_dt > meeting_start)
        total += 1
        if status == "attended":
            attended += 1
        records.append({
            "meeting_num": mt.meeting_num,
            "nefnd_id": mt.nefnd_id,
            "nefnd_name": nefnd_name_map.get(int(mt.nefnd_id)) if mt.nefnd_id is not None else None,
            "time": dt_val,
            "status": status,
            "fundargerd": _fundargerd_link(mt.raw_xml),
            "arrival": arrival_dt,
            "leave_note": leave_note,
            "absence_reason": absence_reason,
            "is_late": is_late,
        })

    records.sort(key=lambda r: r["time"])
    pct = (attended / total * 100) if total else None

    # --- Vote expectations for this member ---
    vote_map: Dict[int, str] = {
        int(vn): vote
        for vn, mid, vote in all_vote_details
        if mid is not None and int(mid) == member_id and vn is not None
    }

    vote_meta: Dict[int, Dict[str, Any]] = {}
    for v in votes_full:
        try:
            num = int(v.attr_atkvaedagreidslunumer)
        except Exception:
            continue
        title = getattr(v, "leaf_mal_malsheiti", None)
        link = f"https://www.althingi.is/thingstorf/thingmalin/atkvaedagreidsla/?nnafnak={num}"
        vote_meta[num] = {"title": title, "link": link}

    sessions: List[Dict[str, Any]] = []
    for vn, t in vote_sessions:
        if vn is None:
            continue
        ts = (parse_date(t) or _parse_iso(t)) if t else None
        meta = vote_meta.get(int(vn), {})
        sessions.append({
            "vote_num": int(vn),
            "time": ts,
            "title": meta.get("title"),
            "link": meta.get("link"),
        })
    sessions.sort(key=lambda s: s["vote_num"])

    vote_expected: List[Dict[str, Any]] = []
    v_attended = v_notified = v_absent = 0
    for vs in sessions:
        ts = vs["time"]
        include = False
        if member_intervals:
            include = _in_intervals(ts, member_intervals) if ts else True
        else:
            include = True
        if not include:
            continue
        vote = vote_map.get(vs["vote_num"])
        status = vote or "fjarverandi"
        if vote in ("já", "nei", "greiðir ekki atkvæði"):
            v_attended += 1
        elif vote == "boðaði fjarvist":
            v_notified += 1
        else:
            v_absent += 1
        vote_expected.append({
            "vote_num": vs["vote_num"],
            "time": ts,
            "status": status,
            "title": vs.get("title"),
            "link": vs.get("link"),
        })
    v_total = len(vote_expected)
    v_pct = (v_attended / v_total * 100) if v_total else None

    return render_template("member_attendance.html",
                           member=member,
                           current_lthing=lthing,
                           attended=attended,
                           total=total,
                           pct=pct,
                           records=records,
                           vote_expected=vote_expected,
                           vote_attended=v_attended,
                           vote_notified=v_notified,
                           vote_absent=v_absent,
                           vote_total=v_total,
                           vote_pct=v_pct)



@bp.route("/speeches")
def speeches():
    Speech = getattr(manual_models, "Speech", None) if manual_models else None
    empty_stats = {
        "longest": [],
        "shortest": [],
        "fastest": [],
        "slowest": [],
        "avg_length": [],
        "avg_speed": [],
        "most_speeches": [],
        "fewest_speeches": [],
    }
    engine = _get_engine()
    with Session(engine) as session:
        lthing = _selected_lthing(session)
        people = session.execute(
            select(models.ThingmannalistiThingmadur).where(models.ThingmannalistiThingmadur.ingest_lthing == lthing)
        ).scalars().all()
        member_map = {}
        for p in people:
            if getattr(p, "attr_id", None) is not None:
                try:
                    member_map[int(p.attr_id)] = p
                except Exception:
                    continue
        if not Speech:
            return render_template("speeches.html",
                                   current_lthing=lthing,
                                   summary=None,
                                   stats=empty_stats,
                                   member_focus=None,
                                   member_options=[],
                                   error="Engin ræðugögn tiltæk.")
        filters = []
        if lthing is not None:
            filters.append(Speech.lthing == lthing)

        total_speeches = session.execute(select(func.count()).where(*filters)).scalar() or 0
        total_words = session.execute(
            select(func.sum(Speech.word_count)).where(*(filters + [Speech.word_count.is_not(None)]))
        ).scalar() or 0
        total_speakers = session.execute(
            select(func.count(func.distinct(Speech.member_id))).where(*(filters + [Speech.member_id.is_not(None)]))
        ).scalar() or 0
        avg_words = (total_words / total_speeches) if total_speeches else None
        avg_wpm_all = session.execute(
            select(func.avg(Speech.words_per_minute)).where(*(filters + [Speech.words_per_minute.is_not(None)]))
        ).scalar()

        minister_map = _minister_map(session, lthing)
        minister_name_to_id = {name: info["id"] for name, info in minister_map.items()}

        def _avatar_url(name: str) -> str:
            safe = quote(name or "")
            return f"https://ui-avatars.com/api/?name={safe}&background=4fd1c5&color=0b1021&bold=true"

        def speech_item(s):
            member = member_map.get(int(s.member_id)) if getattr(s, "member_id", None) is not None else None
            title = getattr(s, "issue_malsheiti", None) or getattr(s, "kind", None) or getattr(s, "umraeda", None)
            member_id = getattr(s, "member_id", None)
            if member_id is None:
                name_key = (getattr(s, "speaker_name", None) or "").strip()
                if name_key in minister_name_to_id:
                    member_id = minister_name_to_id[name_key]
            display_name = getattr(member, "leaf_nafn", None) or getattr(s, "speaker_name", None) or "Ónafngreindur"
            photo_url = None
            if member_id is not None:
                photo_url = f"https://www.althingi.is/myndir/mynd/thingmenn/{member_id}/org/mynd.jpg"
            if not photo_url:
                photo_url = _avatar_url(display_name)
            return {
                "id": s.id,
                "member_id": member_id,
                "speaker": display_name,
                "party": getattr(member, "leaf_skammstofun", None),
                "photo": photo_url,
                "words": getattr(s, "word_count", None),
                "wpm": getattr(s, "words_per_minute", None),
                "duration": _fmt_seconds(getattr(s, "duration_seconds", None)),
                "date": getattr(s, "date", None),
                "title": title,
                "html": getattr(s, "html_url", None),
                "xml": getattr(s, "xml_url", None),
            }

        longest = session.execute(
            select(Speech).where(*(filters + [Speech.word_count.is_not(None)])).order_by(Speech.word_count.desc()).limit(5)
        ).scalars().all()
        shortest = session.execute(
            select(Speech).where(*(filters + [Speech.word_count.is_not(None), Speech.word_count > 0]))
            .order_by(Speech.word_count.asc()).limit(5)
        ).scalars().all()
        fastest = session.execute(
            select(Speech).where(*(filters + [Speech.words_per_minute.is_not(None)])).order_by(Speech.words_per_minute.desc()).limit(5)
        ).scalars().all()
        slowest = session.execute(
            select(Speech).where(*(filters + [Speech.words_per_minute.is_not(None), Speech.words_per_minute > 0]))
            .order_by(Speech.words_per_minute.asc()).limit(5)
        ).scalars().all()

        member_rows = session.execute(
            select(
                Speech.member_id,
                func.count().label("count"),
                func.avg(Speech.word_count).label("avg_words"),
                func.avg(Speech.words_per_minute).label("avg_wpm"),
                func.sum(Speech.word_count).label("sum_words"),
                func.max(Speech.speaker_name).label("speaker_name"),
            ).where(*(filters + [Speech.member_id.is_not(None)])).group_by(Speech.member_id)
        ).all()

        member_stats = []
        for mid, count, avg_w, avg_wpm, sum_w, speaker_name in member_rows:
            if mid is None:
                continue
            member = member_map.get(int(mid))
            display_name = getattr(member, "leaf_nafn", None) or speaker_name or f"Þingmaður {mid}"
            photo_url = f"https://www.althingi.is/myndir/mynd/thingmenn/{int(mid)}/org/mynd.jpg"
            member_stats.append({
                "member_id": int(mid),
                "name": display_name,
                "party": getattr(member, "leaf_skammstofun", None),
                "photo": photo_url,
                "count": count or 0,
                "avg_words": avg_w or 0,
                "avg_wpm": avg_wpm or 0,
                "sum_words": sum_w or 0,
            })
        member_stats.sort(key=lambda m: icelandic_sort_key(m["name"]))

        eligible_member_ids = None
        if manual_models and hasattr(manual_models, "MemberSeat"):
            try:
                MemberSeat = manual_models.MemberSeat
                seat_types = session.execute(
                    select(MemberSeat.member_id, MemberSeat.type)
                    .where(MemberSeat.lthing == lthing, MemberSeat.type.is_not(None))
                ).all()
                eligible_member_ids = {
                    int(mid)
                    for (mid, mtype) in seat_types
                    if mid is not None and (mtype or "").strip().lower() != "varamaður"
                }
            except Exception:
                eligible_member_ids = None

        def top_by(key: str, reverse: bool = True) -> List[Dict[str, Any]]:
            return sorted(member_stats, key=lambda m: (m.get(key) or 0, icelandic_sort_key(m["name"])), reverse=reverse)[:5]

        def top_by_eligible(key: str, reverse: bool = True) -> List[Dict[str, Any]]:
            if not eligible_member_ids:
                return top_by(key, reverse=reverse)
            eligible = [m for m in member_stats if m["member_id"] in eligible_member_ids]
            return sorted(eligible, key=lambda m: (m.get(key) or 0, icelandic_sort_key(m["name"])), reverse=reverse)[:5]

        stats = {
            "longest": [speech_item(s) for s in longest],
            "shortest": [speech_item(s) for s in shortest],
            "fastest": [speech_item(s) for s in fastest],
            "slowest": [speech_item(s) for s in slowest],
            "avg_length": top_by_eligible("avg_words"),
            "avg_speed": top_by_eligible("avg_wpm"),
            "most_speeches": top_by("count"),
            "fewest_speeches": top_by_eligible("count", reverse=False),
        }

        member_id = request.args.get("member_id", type=int)
        member_focus = None
        if member_id:
            speech_rows = session.execute(
                select(Speech).where(*(filters + [Speech.member_id == member_id]))
                .order_by(Speech.start_time.desc())
                .limit(50)
            ).scalars().all()
            if speech_rows:
                total_w = sum(getattr(s, "word_count", 0) or 0 for s in speech_rows)
                wpm_vals = [getattr(s, "words_per_minute", None) for s in speech_rows if getattr(s, "words_per_minute", None) is not None]
                member_focus = {
                    "member": member_map.get(int(member_id)),
                    "member_id": member_id,
                    "count": len(speech_rows),
                    "avg_words": (total_w / len(speech_rows)) if speech_rows else None,
                    "avg_wpm": (sum(wpm_vals) / len(wpm_vals)) if wpm_vals else None,
                    "speeches": [speech_item(s) for s in speech_rows],
                }

        member_options = [
            {"member_id": m["member_id"], "name": m["name"], "party": m["party"], "count": m["count"]}
            for m in member_stats
        ]

        # per-löggjafarþing time-series for charts
        chart_labels: List[int] = []
        chart_series = {
            "longest_words": [],
            "shortest_words": [],
            "fastest_wpm": [],
            "slowest_wpm": [],
            "avg_words": [],
            "total_words": [],
        }
        agg_rows = session.execute(
            select(
                Speech.lthing,
                func.max(Speech.word_count),
                func.min(func.nullif(Speech.word_count, 0)),
                func.max(Speech.words_per_minute),
                func.min(func.nullif(Speech.words_per_minute, 0)),
                func.avg(Speech.word_count),
                func.sum(Speech.word_count),
            )
            .where(Speech.lthing.is_not(None))
            .group_by(Speech.lthing)
        ).all()
        for row in sorted(agg_rows, key=lambda r: r[0]):
            lthing_val = row[0]
            if lthing_val is None:
                continue
            if int(lthing_val) < 115:
                continue
            total_words = int(row[6] or 0)
            if total_words <= 0:
                continue
            chart_labels.append(int(lthing_val))
            chart_series["longest_words"].append(int(row[1] or 0))
            chart_series["shortest_words"].append(int(row[2] or 0))
            chart_series["fastest_wpm"].append(float(row[3] or 0))
            chart_series["slowest_wpm"].append(float(row[4] or 0))
            chart_series["avg_words"].append(float(row[5] or 0))
            chart_series["total_words"].append(total_words)

    error = None
    if total_speeches == 0:
        error = "Engar ræður skráðar fyrir þetta löggjafarþing."
    summary = {
        "total": total_speeches,
        "total_speakers": total_speakers,
        "avg_words": avg_words,
        "avg_wpm": avg_wpm_all,
    }
    return render_template(
        "speeches.html",
        current_lthing=lthing,
        summary=summary,
        stats=stats,
        member_focus=member_focus,
        member_options=member_options,
        chart_labels=chart_labels,
        chart_series=chart_series,
        error=error,
    )


@bp.route("/questions")
def questions():
    engine = _get_engine()
    with Session(engine) as session:
        lthing = _selected_lthing(session)
        if lthing is None:
            lthing = _current_lthing(session)

        people = session.execute(
            select(models.ThingmannalistiThingmadur)
            .where(models.ThingmannalistiThingmadur.ingest_lthing == lthing)
        ).scalars().all()
        member_map = {}
        for p in people:
            if getattr(p, "attr_id", None) is not None:
                try:
                    member_map[int(p.attr_id)] = p
                except Exception:
                    continue

        seats = session.execute(
            select(MemberSeat).where(MemberSeat.lthing == lthing)
        ).scalars().all() if manual_models else []

        def to_regular_days(days_total: int) -> int:
            return int((days_total * 5) // 7)

        seat_by_member: Dict[int, MemberSeat] = {}
        for seat in seats:
            if seat.member_id is None:
                continue
            try:
                mid = int(seat.member_id)
            except Exception:
                continue
            existing = seat_by_member.get(mid)
            inn_dt = parse_date(seat.inn) or dt.datetime.min
            existing_dt = parse_date(existing.inn) if existing else None
            if existing is None or (existing_dt or dt.datetime.min) < inn_dt:
                seat_by_member[mid] = seat

        party_by_member: Dict[int, str] = {}
        for m in people:
            if m.attr_id is None:
                continue
            try:
                mid = int(m.attr_id)
            except Exception:
                continue
            seat = seat_by_member.get(mid)
            party = None
            if seat and seat.party_name:
                party = seat.party_name
            elif seat and seat.party_id:
                party = str(seat.party_id)
            elif m.leaf_skammstofun:
                party = m.leaf_skammstofun
            if party:
                party_by_member[mid] = party

        issues = session.execute(
            select(models.ThingmalalistiMal)
            .where(models.ThingmalalistiMal.ingest_lthing == lthing)
        ).scalars().all()

        question_issues = []
        for issue in issues:
            itype = canonical_issue_type(
                getattr(issue, "leaf_malstegund_heiti2", None) or getattr(issue, "leaf_malstegund_heiti", None)
            )
            if itype and itype.casefold() == WRITTEN_QUESTION_LABEL.casefold():
                question_issues.append(issue)

        docs_thingskjal = session.execute(
            select(models.ThingskjalalistiThingskjal)
            .where(models.ThingskjalalistiThingskjal.ingest_lthing == lthing)
        ).scalars().all()
        dates = []
        for doc in docs_thingskjal:
            attach_flutningsmenn(doc)
            dtv = parse_date(getattr(doc, "leaf_utbyting", None))
            if dtv:
                dates.append(dtv.date())
        session_end = max(dates) if dates else dt.date.today()
        docs_by_nr: Dict[int, Any] = {}
        for doc in docs_thingskjal:
            if doc.attr_skjalsnumer is None:
                continue
            try:
                docs_by_nr[int(doc.attr_skjalsnumer)] = doc
            except Exception:
                continue

        issue_docs = session.execute(
            select(IssueDocument).where(IssueDocument.lthing == lthing)
        ).scalars().all() if manual_models and hasattr(manual_models, "IssueDocument") else []

        docs_by_mal: Dict[tuple, List[Any]] = defaultdict(list)
        for d in issue_docs:
            stored = docs_by_nr.get(d.skjalnr or -1)
            proxy = type("DocProxy", (), {
                "leaf_skjalategund": getattr(stored, "leaf_skjalategund", None),
                "leaf_utbyting": getattr(stored, "leaf_utbyting", None) or getattr(d, "utbyting", None),
                "leaf_athugasemd": getattr(stored, "leaf_athugasemd", None),
                "_flutningsmenn": getattr(stored, "_flutningsmenn", []),
            })
            docs_by_mal[(d.malnr, d.malflokkur)].append(proxy)

        party_filter = request.args.get("party")
        status_filter = request.args.get("status")

        entries = []
        available_parties: set[str] = set()
        for issue in question_issues:
            if issue.attr_malsnumer is None:
                continue
            mal_key = (issue.attr_malsnumer, getattr(issue, "attr_malsflokkur", None))
            docs_for_issue = (
                docs_by_mal.get(mal_key)
                or docs_by_mal.get((issue.attr_malsnumer, None))
                or docs_by_mal.get((issue.attr_malsnumer, ""))
                or []
            )

            primary_mid = None
            for doc in docs_for_issue:
                mid = flutningsmenn_primary_id(doc)
                if mid is not None:
                    primary_mid = int(mid)
                    break
            member = member_map.get(primary_mid) if primary_mid is not None else None
            party = party_by_member.get(primary_mid) if primary_mid is not None else None
            if party:
                available_parties.add(party)

            question_dt = None
            answer_dt = None
            for doc in docs_for_issue:
                stype = (getattr(doc, "leaf_skjalategund", "") or "").lower()
                utb = parse_date(getattr(doc, "leaf_utbyting", None))
                is_question_doc = ("fsp" in stype) or ("fyrirspurn" in stype)
                is_answer_doc = ("svar" in stype) and not is_question_doc
                if is_question_doc and utb:
                    if question_dt is None or utb < question_dt:
                        question_dt = utb
                if is_answer_doc:
                    attn = getattr(doc, "leaf_athugasemd", None) or ""
                    cand_dt = prefer_athugasemd_date(attn) or utb
                    if cand_dt and (answer_dt is None or cand_dt < answer_dt):
                        answer_dt = cand_dt
            status = ANSWER_STATUS_SVARAD if answer_dt else ANSWER_STATUS_OSVARAD

            if party_filter and party_filter != party:
                continue
            if status_filter and status_filter != status:
                continue

            latency = None
            regular_days = None
            if question_dt:
                today = dt.date.today()
                if answer_dt:
                    end_dt = answer_dt.date()
                else:
                    end_dt = today
                delta_days = max((end_dt - question_dt.date()).days, 0)
                regular_days = to_regular_days(delta_days)
                latency = business_days_between(question_dt.date(), end_dt)
            else:
                latency = None

            entries.append({
                "malnr": issue.attr_malsnumer,
                "title": issue.leaf_malsheiti,
                "party": party,
                "member": member,
                "status": status,
                "question_date": question_dt.date() if question_dt else None,
                "answer_date": answer_dt.date() if answer_dt else None,
                "latency": latency,
                "regular_days": regular_days,
                "html": issue.leaf_html,
                "xml": issue.leaf_xml,
            })

        entries.sort(key=lambda e: (e["question_date"] or dt.date.min), reverse=True)
        party_options = sorted(available_parties, key=icelandic_sort_key)

        def to_regular_days(business_days: int) -> int:
            return int((business_days * 5) // 7)

        answered_latencies = [to_regular_days(e["latency"]) for e in entries if e["status"] == ANSWER_STATUS_SVARAD and e["latency"] is not None]
        all_latencies = [to_regular_days(e["latency"]) for e in entries if e["latency"] is not None]
        avg_answered = sum(answered_latencies) / len(answered_latencies) if answered_latencies else None
        avg_all = sum(all_latencies) / len(all_latencies) if all_latencies else None

    return render_template(
        "questions.html",
        current_lthing=lthing,
        questions=entries,
        party_options=party_options,
        party_filter=party_filter or "",
        status_filter=status_filter or "",
        avg_answered=avg_answered,
        avg_all=avg_all,
        error=None if entries else "Engar fyrirspurnir fundust fyrir þetta þing.",
    )


@bp.route("/speeches/member/<int:member_id>")
def member_speeches(member_id: int):
    Speech = getattr(manual_models, "Speech", None) if manual_models else None
    engine = _get_engine()
    with Session(engine) as session:
        lthing = _selected_lthing(session)
        member = None
        name_filter = None
        if member_id < 0:
            minister_map = _minister_map(session, lthing)
            minister_by_id = {info["id"]: {"name": name, **info} for name, info in minister_map.items()}
            info = minister_by_id.get(member_id)
            if info:
                member = type("MemberProxy", (), {
                    "attr_id": member_id,
                    "leaf_nafn": info["name"],
                    "leaf_skammstofun": None,
                    "is_pseudo": True,
                })
                name_filter = info["name"]
        else:
            people = session.execute(
                select(models.ThingmannalistiThingmadur).where(models.ThingmannalistiThingmadur.ingest_lthing == lthing)
            ).scalars().all()
            member_map = {}
            for p in people:
                if getattr(p, "attr_id", None) is not None:
                    try:
                        member_map[int(p.attr_id)] = p
                    except Exception:
                        continue
            member = member_map.get(int(member_id))
        if not Speech:
            return render_template(
                "member_speeches.html",
                current_lthing=lthing,
                member=member,
                member_id=member_id,
                summary=None,
                speeches=[],
                limit=0,
                error="Engin ræðugögn tiltæk.",
            )

        if name_filter:
            filters = [Speech.member_id.is_(None), Speech.speaker_name == name_filter]
        else:
            filters = [Speech.member_id == member_id]
        if lthing is not None:
            filters.append(Speech.lthing == lthing)

        total_speeches = session.execute(
            select(func.count()).where(*filters)
        ).scalar() or 0
        total_words = session.execute(
            select(func.sum(Speech.word_count)).where(*(filters + [Speech.word_count.is_not(None)]))
        ).scalar() or 0
        avg_wpm = session.execute(
            select(func.avg(Speech.words_per_minute)).where(*(filters + [Speech.words_per_minute.is_not(None)]))
        ).scalar()

        limit = request.args.get("limit", type=int) or 200
        limit = max(1, min(limit, 500))
        speech_rows = session.execute(
            select(Speech).where(*filters).order_by(Speech.start_time.desc()).limit(limit)
        ).scalars().all()

        def speech_item(s):
            title = getattr(s, "issue_malsheiti", None) or getattr(s, "kind", None) or getattr(s, "umraeda", None)
            return {
                "id": s.id,
                "words": getattr(s, "word_count", None),
                "wpm": getattr(s, "words_per_minute", None),
                "duration": _fmt_seconds(getattr(s, "duration_seconds", None)),
                "date": getattr(s, "date", None),
                "title": title,
                "html": getattr(s, "html_url", None),
                "xml": getattr(s, "xml_url", None),
            }

        summary = {
            "total": total_speeches,
            "total_words": total_words,
            "avg_words": (total_words / total_speeches) if total_speeches else None,
            "avg_wpm": avg_wpm,
        }

    return render_template(
        "member_speeches.html",
        current_lthing=lthing,
        member=member,
        member_id=member_id,
        summary=summary,
        speeches=[speech_item(s) for s in speech_rows],
        limit=limit,
        error=None,
    )


@bp.route("/votes/report")
def vote_report():
    """
    Per-member vote report: expected vote sessions (based on inn/út þingseta) and recorded vote/absence.
    """
    engine = _get_engine()
    with Session(engine) as session:
        lthing = _selected_lthing(session)
        people = session.execute(
            select(models.ThingmannalistiThingmadur)
            .where(models.ThingmannalistiThingmadur.ingest_lthing == lthing)
            .order_by(models.ThingmannalistiThingmadur.leaf_nafn)
        ).scalars().all()
        seats = session.execute(
            select(MemberSeat).where(MemberSeat.lthing == lthing)
        ).scalars().all() if manual_models else []

        vote_sessions = []
        if manual_models and hasattr(manual_models, "VoteSession"):
            vote_sessions = session.execute(
                select(manual_models.VoteSession.vote_num, manual_models.VoteSession.time)
                .where(manual_models.VoteSession.lthing == lthing)
            ).all()
        if not vote_sessions:
            vote_sessions = session.execute(
                select(
                    models.AtkvaedagreidslurAtkvaedagreidsla.attr_atkvaedagreidslunumer,
                    models.AtkvaedagreidslurAtkvaedagreidsla.leaf_timi,
                ).where(models.AtkvaedagreidslurAtkvaedagreidsla.ingest_lthing == lthing)
            ).all()

        all_vote_details = session.execute(
            select(VoteDetail.vote_num, VoteDetail.voter_id, VoteDetail.vote)
            .where(VoteDetail.lthing == lthing)
        ).all() if manual_models else []

    intervals_by_member = _effective_intervals(seats)

    vote_map_by_member: Dict[int, Dict[int, str]] = defaultdict(dict)
    for vn, mid, vote in all_vote_details:
        if vn is None or mid is None:
            continue
        try:
            vote_map_by_member[int(mid)][int(vn)] = vote
        except Exception:
            continue

    vote_meta: Dict[int, Dict[str, Any]] = {}
    for v in votes_full:
        try:
            num = int(v.attr_atkvaedagreidslunumer)
        except Exception:
            continue
        title = getattr(v, "leaf_mal_malsheiti", None)
        link = (
            getattr(v, "leaf_nanar_html", None)
            or getattr(v, "leaf_mal_html", None)
            or getattr(v, "leaf_nanar_xml", None)
            or getattr(v, "leaf_mal_xml", None)
            or getattr(v, "leaf_thingskjal_slod_xml", None)
        )
        vote_meta[num] = {
            "title": title,
            "html": link,
        }

    sessions: List[Dict[str, Any]] = []
    for vn, t in vote_sessions:
        if vn is None:
            continue
        ts = (parse_date(t) or _parse_iso(t)) if t else None
        meta = vote_meta.get(int(vn), {})
        sessions.append({
            "vote_num": int(vn),
            "time": ts,
            "title": meta.get("title"),
            "link": meta.get("html"),
        })
    sessions.sort(key=lambda s: s["vote_num"])

    report = []
    for p in people:
        if p.attr_id is None:
            continue
        mid = int(p.attr_id)
        intervals = intervals_by_member.get(mid, [])
        if not intervals:
            continue
        expected = []
        attended = 0
        notified = 0
        absent = 0
        for vs in sessions:
            ts = vs["time"]
            include = False
            if intervals:
                if ts:
                    include = _in_intervals(ts, intervals)
                else:
                    include = True  # no timestamp on vote -> include to be safe
            if not include:
                continue
            vote = vote_map_by_member.get(mid, {}).get(vs["vote_num"])
            status = vote or "fjarverandi"
            if vote in ("já", "nei", "greiðir ekki atkvæði"):
                attended += 1
            elif vote == "boðaði fjarvist":
                notified += 1
            else:
                absent += 1
            expected.append({
                "vote_num": vs["vote_num"],
                "time": ts,
                "status": status,
                "title": vs.get("title"),
                "link": vs.get("link"),
            })
        if not expected:
            continue
        total = len(expected)
        pct = attended / total * 100 if total else None
        report.append({
            "member": p,
            "expected": expected,
            "attended": attended,
            "notified": notified,
            "absent": absent,
            "total": total,
            "pct": pct,
        })

    report.sort(key=lambda r: icelandic_sort_key(r["member"].leaf_nafn or ""))

    return render_template(
        "vote_report.html",
        report=report,
        current_lthing=lthing,
    )


@bp.route("/committees")
def committees():
    engine = _get_engine()
    with Session(engine) as session:
        lthing = _selected_lthing(session)
        committees = session.execute(
            select(models.NefndirNefnd)
            .where(models.NefndirNefnd.ingest_lthing == lthing)
            .order_by(models.NefndirNefnd.leaf_heiti)
        ).scalars().all()
        mal_rows = session.execute(
            select(models.ThingmalalistiMal).where(models.ThingmalalistiMal.ingest_lthing == lthing)
        ).scalars().all()
        votes = session.execute(
            select(models.AtkvaedagreidslurAtkvaedagreidsla)
            .where(models.AtkvaedagreidslurAtkvaedagreidsla.ingest_lthing == lthing)
        ).scalars().all()
        meeting_rows = session.execute(
            select(manual_models.CommitteeMeeting).where(manual_models.CommitteeMeeting.lthing == lthing)
        ).scalars().all() if manual_models else []
        attendance_rows = session.execute(
            text(
                """
                SELECT cm.nefnd_id, cm.start_time, ca.member_id, ca.status
                FROM committee_attendance ca
                JOIN committee_meeting cm ON ca.meeting_id = cm.id
                WHERE cm.lthing = :lt
                """
            ),
            {"lt": lthing},
        ).fetchall()
        people = session.execute(
            select(models.ThingmannalistiThingmadur).where(models.ThingmannalistiThingmadur.ingest_lthing == lthing)
        ).scalars().all()
        nefnd_members_rows = session.execute(
            select(NefndMember).where(NefndMember.lthing == lthing)
        ).scalars().all()
        member_seats = session.execute(
            select(manual_models.MemberSeat).where(manual_models.MemberSeat.lthing == lthing)
        ).scalars().all() if manual_models else []

    member_name: Dict[int, str] = {}
    for p in people:
        if p.attr_id is not None:
            member_name[int(p.attr_id)] = p.leaf_nafn or ""

    mal_by_key = {(m.attr_malsnumer, getattr(m, "attr_malsflokkur", None)): m for m in mal_rows if m.attr_malsnumer is not None}

    def norm(name: str) -> str:
        return (name or "").strip().lower()

    def _parse_dt(val: Optional[str]) -> Optional[dt.datetime]:
        if not val:
            return None
        try:
            return dt.datetime.fromisoformat(val)
        except Exception:
            return parse_date(val)

    meeting_dates_by_nefnd: Dict[int, List[dt.datetime]] = defaultdict(list)
    for m in meeting_rows:
        if m.nefnd_id is None:
            continue
        dt_val = _parse_dt(m.start_time)
        if dt_val:
            meeting_dates_by_nefnd[int(m.nefnd_id)].append(dt_val)

    seat_intervals: Dict[int, List[Tuple[Optional[dt.date], Optional[dt.date]]]] = defaultdict(list)
    for seat in member_seats:
        if seat.member_id is None:
            continue
        inn_dt = parse_date(seat.inn)
        ut_dt = parse_date(seat.ut)
        seat_intervals[int(seat.member_id)].append((inn_dt.date() if inn_dt else None, ut_dt.date() if ut_dt else None))

    def active_on(member_id: int, nefnd_id: int, when: dt.date) -> bool:
        # Must be active both in Alþingi seat (seat_intervals) and committee seat (committee_intervals)
        intervals_a = seat_intervals.get(member_id, [])
        intervals_c = committee_intervals.get(nefnd_id, {}).get(member_id, [])
        def match(intervals):
            if not intervals:
                return True
            for inn, ut in intervals:
                if inn and when < inn:
                    continue
                if ut and when > ut:
                    continue
                return True
            return False
        return match(intervals_a) and match(intervals_c)

    attendance_by_nm: Dict[tuple, Dict[str, int]] = {}
    members_by_nefnd: Dict[int, List[Any]] = defaultdict(list)
    seen_members_nefnd: Dict[int, set] = defaultdict(set)
    official_members: Dict[int, set] = defaultdict(set)
    committee_intervals: Dict[int, Dict[int, List[Tuple[Optional[dt.date], Optional[dt.date]]]]] = defaultdict(lambda: defaultdict(list))

    def role_rank(role: Optional[str]) -> int:
        if not role:
            return 99
        lr = role.lower().replace(".", "").strip()
        if "formaður" in lr and "vara" not in lr:
            return 0
        if lr.startswith("1 varaformaður") or "1. varaformaður" in lr or "1 varaformadur" in lr:
            return 1
        if lr.startswith("2 varaformaður") or "2. varaformaður" in lr or "2 varaformadur" in lr:
            return 2
        if "nefndarmaður" in lr:
            return 3
        if "áheyrnarfulltrú" in lr:
            return 4
        if "varamaður" in lr:
            return 5
        return 99

    # Pick latest seat (by start date, then latest end/open, then best rank) per member per committee; capture committee-specific intervals
    best_roles: Dict[int, Dict[int, Tuple[int, str, str, Optional[dt.date], Optional[dt.date]]]] = defaultdict(dict)  # nefnd_id -> member_id -> (rank, role, name, inn_date, ut_date)
    for nm in nefnd_members_rows:
        if nm.nefnd_id is None or nm.member_id is None:
            continue
        nid = int(nm.nefnd_id)
        mid = int(nm.member_id)
        rank = role_rank(nm.role)
        inn_dt = parse_date(nm.inn)
        ut_dt = parse_date(nm.ut)
        inn_date = inn_dt.date() if inn_dt else None
        ut_date = ut_dt.date() if ut_dt else None
        committee_intervals[nid][mid].append((inn_date, ut_date))
        current = best_roles[nid].get(mid)
        new_key = (inn_date or dt.date.min, ut_date or dt.date.max, -rank)
        curr_key = (current[3] or dt.date.min, current[4] or dt.date.max, -(current[0])) if current else (dt.date.min, dt.date.min, 0)
        if current is None or new_key > curr_key or (new_key == curr_key and rank < current[0]):
            best_roles[nid][mid] = (rank, nm.role or "", nm.name or member_name.get(mid, f"Þingmaður {mid}"), inn_date, ut_date)

    latest_meeting_date: Dict[int, Optional[dt.date]] = {
        nid: max([d.date() for d in dates], default=None) for nid, dates in meeting_dates_by_nefnd.items()
    }

    def interval_covers(member_id: int, nefnd_id: int, when: Optional[dt.date]) -> bool:
        if when is None:
            return True
        intervals = committee_intervals.get(nefnd_id, {}).get(member_id, [])
        if not intervals:
            return True
        for inn, ut in intervals:
            if inn and when < inn:
                continue
            if ut and when > ut:
                continue
            return True
        return False

    # Seed members list and official set from best role (only if they cover latest meeting date)
    for nid, members in best_roles.items():
        for mid, (rank, role, name_val, _, _) in members.items():
            if not interval_covers(mid, nid, latest_meeting_date.get(nid)):
                continue
            is_official = rank <= 3  # formaður, 1./2. varaformaður, nefndarmaður
            if is_official:
                official_members[nid].add(mid)
            if mid not in seen_members_nefnd[nid]:
                seen_members_nefnd[nid].add(mid)
                members_by_nefnd[nid].append(type("NefndMemberProxy", (), {
                    "member_id": mid,
                    "name": name_val,
                    "role": role,
                }))

    appearances_any: Dict[int, int] = defaultdict(int)
    present_by_meeting: Dict[Tuple[int, dt.date], set] = defaultdict(set)

    for nefnd_id, start_time, member_id, status in attendance_rows:
        if nefnd_id is None or member_id is None:
            continue
        dt_val = _parse_dt(start_time)
        if status in ("present", "proxy_present"):
            appearances_any[int(member_id)] += 1
            if dt_val:
                present_by_meeting[(int(nefnd_id), dt_val.date())].add(int(member_id))

    # Build official attendance (only formal members, not varamaður/áheyrnarfulltrúi)
    for nid, mids in official_members.items():
        dates = meeting_dates_by_nefnd.get(nid, [])
        for mid in mids:
            total = 0
            attended = 0
            for d in dates:
                if not active_on(mid, nid, d.date()):
                    continue
                total += 1
                if mid in present_by_meeting.get((nid, d.date()), set()):
                    attended += 1
            attendance_by_nm[(nid, mid)] = {
                "attended": attended,
                "total": total,
                "appearances": appearances_any.get(mid, 0),
            }
            if mid not in seen_members_nefnd[nid]:
                seen_members_nefnd[nid].add(mid)
                members_by_nefnd[nid].append(type("NefndMemberProxy", (), {
                    "member_id": mid,
                    "name": member_name.get(mid, f"Þingmaður {mid}"),
                    "role": None,
                }))

    # Order members within each committee by role priority then name
    for nid, lst in members_by_nefnd.items():
        lst.sort(key=lambda x: (role_rank(getattr(x, "role", None)), icelandic_sort_key(getattr(x, "name", ""))))

    mal_by_nefnd_name: Dict[str, set] = defaultdict(set)
    for v in votes:
        if v.leaf_tegund and "gengur" in v.leaf_tegund:
            if v.leaf_til:
                key = (v.attr_malsnumer, v.attr_malsflokkur)
                mal_by_nefnd_name[norm(v.leaf_til)].add(key)

    cache_dir = _cache_dir()
    issue_meta_cache: Dict[str, tuple] = {}
    issues_by_nefnd: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for nefnd in committees:
        key_name = norm(nefnd.leaf_heiti)
        mal_keys = mal_by_nefnd_name.get(key_name, set())
        for mk in mal_keys:
            mal = mal_by_key.get(mk)
            if mal:
                status, umsagn_count = _cached_issue_meta(mal.leaf_xml, cache_dir, issue_meta_cache)
                issues_by_nefnd[int(nefnd.attr_id)].append({
                    "malnr": mk[0],
                    "title": mal.leaf_malsheiti,
                    "malflokkur": mk[1],
                    "html": mal.leaf_html,
                    "xml": mal.leaf_xml,
                    "status": status,
                    "umsagn_count": umsagn_count,
                    "umsagn_url": f"https://www.althingi.is/thingstorf/thingmalin/erindi/{lthing}/{mk[0]}/?ltg={lthing}&mnr={mk[0]}",
                })

    # Already sorted by role then name; do not resort by name only.
    for lst in issues_by_nefnd.values():
        lst.sort(key=lambda x: x["malnr"])

    error = None
    if not committees:
        error = "Engar nefndir fundust fyrir þetta löggjafarþing."
    return render_template("committees.html", committees=committees, current_lthing=lthing,
                           members_by_nefnd=members_by_nefnd, issues_by_nefnd=issues_by_nefnd,
                           attendance_by_nm=attendance_by_nm, error=error)


def register(app):
    app.register_blueprint(bp)


@bp.route("/agenda")
def agenda():
    engine = _get_engine()
    try:
        resp = requests.get("https://www.althingi.is/altext/xml/dagskra/thingfundur/", timeout=10)
        resp.raise_for_status()
        agenda_xml = resp.text
    except Exception:
        agenda_xml = None

    with Session(engine) as session:
        lthing = _selected_lthing(session)
        mal_rows = session.execute(
            select(models.ThingmalalistiMal).where(models.ThingmalalistiMal.ingest_lthing == lthing)
        ).scalars().all()
        docs = session.execute(
            select(IssueDocument).where(IssueDocument.lthing == lthing)
        ).scalars().all() if manual_models and hasattr(manual_models, "IssueDocument") else []
        votes = session.execute(
            select(models.AtkvaedagreidslurAtkvaedagreidsla)
            .where(models.AtkvaedagreidslurAtkvaedagreidsla.ingest_lthing == lthing)
        ).scalars().all()
        minister_map = _minister_map(session, lthing)

    docs_by_mal: Dict[tuple, List[Any]] = defaultdict(list)
    for d in docs:
        if d.malnr is not None:
            key = (int(d.malnr), getattr(d, "malflokkur", None))
            docs_by_mal[key].append(d)

    votes_by_mal: Dict[int, List[Any]] = defaultdict(list)
    for v in votes:
        if v.attr_malsnumer is not None:
            votes_by_mal[int(v.attr_malsnumer)].append(v)

    mal_map = {(int(m.attr_malsnumer), getattr(m, "attr_malsflokkur", None)): m for m in mal_rows if m.attr_malsnumer is not None}

    agenda_items = []
    fund_info = {}
    fund_lthing = None
    error = None
    if agenda_xml:
        try:
            root = ET.fromstring(agenda_xml)
            fund = root.find(".//þingfundur") or root.find(".//thingfundur")
            if fund is not None:
                try:
                    fund_lthing = int(fund.attrib.get("þingnúmer") or fund.attrib.get("thingnumer") or fund.attrib.get("thingnúmer") or fund.attrib.get("þingnumer"))
                except Exception:
                    fund_lthing = None
                fund_info = {
                    "fundarheiti": (fund.findtext("fundarheiti") or "").strip(),
                    "dagurtimi": (fund.findtext(".//dagurtími") or fund.findtext(".//dagurtimi") or "").strip(),
                }
                for li in fund.findall(".//dagskrárliður"):
                    num = li.attrib.get("númer") or li.attrib.get("numer")
                    mal_elem = li.find("mál") or li.find("mal")
                    malnr = None
                    malflokkur = None
                    title = None
                    if mal_elem is not None:
                        try:
                            malnr = int(mal_elem.attrib.get("málsnúmer") or mal_elem.attrib.get("malsnumer") or mal_elem.attrib.get("malnumer"))
                        except Exception:
                            malnr = None
                        malflokkur = mal_elem.attrib.get("málsflokkur") or mal_elem.attrib.get("malsflokkur")
                        title = mal_elem.findtext("málsheiti") or mal_elem.findtext("malsheiti")
                    mal_key = (malnr, malflokkur)
                    matched = mal_map.get(mal_key)
                    agenda_items.append({
                        "number": num,
                        "malnr": malnr,
                        "malflokkur": malflokkur,
                        "title": title or (matched.leaf_malsheiti if matched else None),
                        "issue": matched,
                        "docs": docs_by_mal.get(mal_key, []),
                        "votes": votes_by_mal.get(malnr, []),
                        "lthing": fund_lthing,
                    })
        except Exception:
            pass

    if lthing is not None and fund_lthing is not None and lthing != fund_lthing:
        agenda_items = []
        error = "Dagskrá er aðeins tiltæk fyrir núverandi þingfund."

    # attach speeches (cached) for linked issues
    cache_dir = _cache_dir()
    speeches_by_mal: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    issue_meta_cache: Dict[str, list] = {}
    name_to_id = {name: info["id"] for name, info in minister_map.items()}
    for item in agenda_items:
        issue = item.get("issue")
        speeches: List[Dict[str, Any]] = []
        primary_xml = issue.leaf_xml if issue and issue.leaf_xml else None
        sources = []
        if primary_xml:
            sources.append(primary_xml)
        # fallback: bmal XML for B category
        if item.get("malnr") and item.get("malflokkur") == "B" and item.get("lthing"):
            sources.append(f"https://www.althingi.is/altext/xml/thingmalalisti/bmal/?lthing={item.get('lthing')}&malnr={item.get('malnr')}")
        for src in sources:
            path = _cache_path_for_url(src, cache_dir)
            if not path.exists():
                try:
                    resp = requests.get(src, timeout=10)
                    resp.raise_for_status()
                    path.write_text(resp.text, encoding="utf-8")
                except Exception:
                    continue
            speeches = _cached_speeches(src, cache_dir, issue_meta_cache, None, name_to_member_id=name_to_id)
            if speeches:
                break
        speeches_by_mal[(item["malnr"], item["malflokkur"])] = speeches

    return render_template("agenda.html",
                           fund_info=fund_info,
                           items=agenda_items,
                           speeches_by_mal=speeches_by_mal,
                           error=error)
