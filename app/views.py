from __future__ import annotations

import datetime as dt
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Blueprint, current_app, render_template
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from urllib.parse import quote

from . import models
try:
    import app.manual_models as manual_models  # type: ignore
except Exception:
    manual_models = None
from .views_helper import (
    current_lthing,
    parse_date,
    icelandic_sort_key,
    flutningsmenn_primary_id,
)
from .manual_models import IssueDocument, VoteDetail, MemberSeat, NefndMember, attach_flutningsmenn

bp = Blueprint("main", __name__)


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
    for inn, ut in intervals:
        if inn and ts < inn:
            continue
        if ut and ts > ut:
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
        if getattr(seat, "type", "") and "varama" in str(seat.type).lower():
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


def _cached_speeches(
    url: Optional[str],
    cache_dir: Path,
    cache: Dict[str, List[Dict[str, Any]]],
    party_by_member: Optional[Dict[int, str]] = None,
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
    with Session(engine) as session:
        issues = session.execute(
            select(models.ThingmalalistiMal).order_by(models.ThingmalalistiMal.attr_malsnumer)
        ).scalars().all()
        lthing = current_lthing(session)
        members = session.execute(
            select(models.ThingmannalistiThingmadur)
        ).scalars().all()
        seats = session.execute(
            select(MemberSeat).where(MemberSeat.lthing == lthing)
        ).scalars().all()

        votes = session.execute(
            select(models.AtkvaedagreidslurAtkvaedagreidsla)
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
        ).scalars().all()
        docs_by_nr: Dict[int, Any] = {}
        for doc in documents:
            attach_flutningsmenn(doc)
            if doc.attr_skjalsnumer is not None:
                docs_by_nr[int(doc.attr_skjalsnumer)] = doc

        docs_by_mal: Dict[int, List[Any]] = defaultdict(list)
        # Pre-fetched issue documents are stored in IssueDocument; join to flutningsmenn via skjalnr
        issue_docs = session.execute(select(IssueDocument)).scalars().all()
        docs_by_mal: Dict[int, List[Any]] = defaultdict(list)
        for d in issue_docs:
            stored = docs_by_nr.get(d.skjalnr or -1)
            docs_by_mal[d.malnr or -1].append(type("DocProxy", (), {
                "leaf_skjalategund": d.skjalategund or (f"Skjal {d.skjalnr}" if d.skjalnr else "Skjal"),
                "leaf_utbyting": d.utbyting,
                "leaf_slod_html": d.slod_html or (getattr(stored, "leaf_slod_html", None) if stored else None),
                "leaf_slod_pdf": d.slod_pdf or (getattr(stored, "leaf_slod_pdf", None) if stored else None),
                "leaf_slod_xml": d.slod_xml or (getattr(stored, "leaf_slod_xml", None) if stored else None),
                "_flutningsmenn": getattr(stored, "_flutningsmenn", []),
                "attr_skjalsnumer": d.skjalnr,
            }))

        for malnr, items in list(docs_by_mal.items()):
            items.sort(key=lambda d: parse_date(getattr(d, "leaf_utbyting", None)) or dt.datetime.min)

    # map member id -> party name for display with speeches
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
    for issue in issues:
        key = issue.attr_malsnumer
        if key is None:
            continue
        speeches_by_mal[int(key)] = _cached_speeches(
            getattr(issue, "leaf_xml", None),
            cache_dir,
            speeches_cache,
            party_by_member=party_by_member,
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
        typ = issue.leaf_malstegund_heiti2 or issue.leaf_malstegund_heiti
        if typ:
            type_by_issue[int(key)] = typ

    # aggregate counts
    party_counts: Dict[str, int] = defaultdict(int)
    type_counts: Dict[str, int] = defaultdict(int)
    issue_parties_map: Dict[int, List[str]] = {}
    for malnr, parties in parties_by_issue.items():
        for p in parties:
            party_counts[p] += 1
        issue_parties_map[malnr] = sorted(parties, key=icelandic_sort_key)
    for malnr, typ in type_by_issue.items():
        type_counts[typ] += 1

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
    )


@bp.route("/members")
def members():
    engine = _get_engine()
    with Session(engine) as session:
        people = session.execute(
            select(models.ThingmannalistiThingmadur).order_by(models.ThingmannalistiThingmadur.leaf_nafn)
        ).scalars().all()
        lthing = current_lthing(session)
        seats = session.execute(
            select(MemberSeat).where(MemberSeat.lthing == lthing)
        ).scalars().all()
        mal_rows = session.execute(
            select(models.ThingmalalistiMal)
        ).scalars().all()
        issue_docs = session.execute(select(IssueDocument)).scalars().all()
        docs_by_nr = session.execute(select(models.ThingskjalalistiThingskjal)).scalars().all()
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
                )
            ).all()
        all_vote_details = session.execute(
            select(VoteDetail.vote_num, VoteDetail.voter_id, VoteDetail.vote)
        ).all()
        vote_nums_with_votes = {vd[0] for vd in all_vote_details if vd[0] is not None}
        vote_sessions = [vs for vs in vote_sessions if vs[0] in vote_nums_with_votes]

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

    # Committee attendance from table populated at ingest; fallback to cached parsing
    attendance_attended: Dict[int, int] = defaultdict(int)
    attendance_total: Dict[int, int] = defaultdict(int)
    used_manual_att = False
    if manual_models and hasattr(manual_models, "CommitteeAttendance"):
        att_rows = session.execute(
            select(manual_models.CommitteeAttendance.member_id,
                   manual_models.CommitteeAttendance.status,
                   manual_models.CommitteeAttendance.meeting_num)
            .where(manual_models.CommitteeAttendance.lthing == lthing)
        ).all()
        for mid, status, _ in att_rows:
            if mid is None:
                continue
            if status in ("present", "proxy_present"):
                attendance_attended[int(mid)] += 1
                attendance_total[int(mid)] += 1
            elif status in ("absent_notified",):
                attendance_total[int(mid)] += 1
        if att_rows:
            used_manual_att = True
    if not used_manual_att:
        cache_dir = _cache_dir()
        if cache_dir.exists():
            for fp in cache_dir.glob("*nefndarfundur*"):
                try:
                    root = ET.parse(fp).getroot()
                    meeting_dt = None
                    for t in root.iter():
                        tag = _strip_tag(t.tag)
                        if tag == "dagurtími" or tag == "dagurtimi":
                            meeting_dt = _parse_iso((t.text or "").strip())
                            break
                        if tag == "dagur":
                            meeting_dt = parse_date((t.text or "").strip())
                    if meeting_dt is None:
                        meeting_dt = parse_date(root.findtext(".//dagur") or "")
                    texti = None
                    for t in root.iter():
                        if _strip_tag(t.tag) == "texti":
                            texti = t.text or ""
                            break
                    if not texti:
                        continue
                    attended, absent = _parse_attendance_from_html(texti, abbr_to_member)
                    for mid in attended:
                        intervals = intervals_by_member.get(mid, [])
                        if not intervals or _in_intervals(meeting_dt, intervals):
                            attendance_attended[mid] += 1
                            attendance_total[mid] += 1
                    for mid in absent:
                        intervals = intervals_by_member.get(mid, [])
                        if not intervals or _in_intervals(meeting_dt, intervals):
                            attendance_total[mid] += 1
                except Exception:
                    continue

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

    def party_for_member(m):
        if getattr(m, "current_seat", None) and m.current_seat.party_name:
            return m.current_seat.party_name
        return m.leaf_skammstofun or "Óflokkað"

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

    return render_template("members.html", parties=parties, current_lthing=lthing)


@bp.route("/committees")
def committees():
    engine = _get_engine()
    with Session(engine) as session:
        committees = session.execute(
            select(models.NefndirNefnd).order_by(models.NefndirNefnd.leaf_heiti)
        ).scalars().all()
        lthing = current_lthing(session)
        members = session.execute(
            select(NefndMember).where(NefndMember.lthing == lthing)
        ).scalars().all()
        mal_rows = session.execute(
            select(models.ThingmalalistiMal)
        ).scalars().all()
        votes = session.execute(
            select(models.AtkvaedagreidslurAtkvaedagreidsla)
        ).scalars().all()

    members_by_nefnd: Dict[int, List[NefndMember]] = defaultdict(list)
    for m in members:
        if m.nefnd_id is not None:
            members_by_nefnd[int(m.nefnd_id)].append(m)

    mal_by_key = {(m.attr_malsnumer, getattr(m, "attr_malsflokkur", None)): m for m in mal_rows if m.attr_malsnumer is not None}

    def norm(name: str) -> str:
        return (name or "").strip().lower()

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

    for lst in members_by_nefnd.values():
        lst.sort(key=lambda m: icelandic_sort_key(m.name or ""))
    for lst in issues_by_nefnd.values():
        lst.sort(key=lambda x: x["malnr"])

    return render_template("committees.html", committees=committees, current_lthing=lthing,
                           members_by_nefnd=members_by_nefnd, issues_by_nefnd=issues_by_nefnd)


def register(app):
    app.register_blueprint(bp)
