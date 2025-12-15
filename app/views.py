from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any, Dict, List

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


@bp.route("/")
def index():
    engine = _get_engine()
    with Session(engine) as session:
        issues = session.execute(
            select(models.ThingmalalistiMal).order_by(models.ThingmalalistiMal.attr_malsnumer)
        ).scalars().all()
        lthing = current_lthing(session)

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

    return render_template(
        "index.html",
        issues=issues,
        votes_by_mal=votes_by_mal,
        docs_by_mal=docs_by_mal,
        vote_counts=counts_by_vote,
        current_lthing=lthing,
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

    issues_by_nefnd: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for nefnd in committees:
        key_name = norm(nefnd.leaf_heiti)
        mal_keys = mal_by_nefnd_name.get(key_name, set())
        for mk in mal_keys:
            mal = mal_by_key.get(mk)
            if mal:
                issues_by_nefnd[int(nefnd.attr_id)].append({
                    "malnr": mk[0],
                    "title": mal.leaf_malsheiti,
                    "malflokkur": mk[1],
                    "html": mal.leaf_html,
                    "xml": mal.leaf_xml,
                })

    for lst in members_by_nefnd.values():
        lst.sort(key=lambda m: icelandic_sort_key(m.name or ""))
    for lst in issues_by_nefnd.values():
        lst.sort(key=lambda x: x["malnr"])

    return render_template("committees.html", committees=committees, current_lthing=lthing,
                           members_by_nefnd=members_by_nefnd, issues_by_nefnd=issues_by_nefnd)


def register(app):
    app.register_blueprint(bp)
