from __future__ import annotations

import datetime as dt
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, current_app, render_template
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from urllib.parse import quote

from . import models
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

    docs_map = {}
    for d in docs_by_nr:
        attach_flutningsmenn(d)
        if d.attr_skjalsnumer is not None:
            docs_map[int(d.attr_skjalsnumer)] = d

    seat_by_member: Dict[int, MemberSeat] = {}
    for seat in seats:
        key = int(seat.member_id)
        existing = seat_by_member.get(key)
        inn_dt = parse_date(seat.inn) or dt.datetime.min
        existing_dt = parse_date(existing.inn) if existing else None
        if existing is None or (existing_dt or dt.datetime.min) < inn_dt:
            seat_by_member[key] = seat

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
