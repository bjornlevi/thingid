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
import importlib
import json
import os
import sys
import time
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import xml.etree.ElementTree as ET
from sqlalchemy import UniqueConstraint, create_engine, text
from sqlalchemy.orm import Session

BASE_CURRENT = "https://www.althingi.is/altext/xml/loggjafarthing/yfirstandandi/"


# -----------------------------
# Helpers
# -----------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

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
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, url: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", url)
        return os.path.join(self.cache_dir, safe)

    def get(self, url: str, retries: int = 4) -> bytes:
        cache_path = self._cache_path(url)
        if cache_path and not self.force and os.path.exists(cache_path):
            age_hours = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))).total_seconds() / 3600.0
            if age_hours < self.max_age_hours:
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

def discover_current_lthing_and_yfirlit(fetcher: Fetcher) -> Tuple[int, Dict[str, str]]:
    # Minimal parse: choose max <þing númer="..."> like in check_data
    root = parse_xml(fetcher.get(BASE_CURRENT), BASE_CURRENT)

    sessions = []
    for e in root.iter():
        if strip_ns(e.tag) == "þing" and e.attrib:
            sessions.append(e)
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

    yf = None
    for c in list(best):
        if strip_ns(c.tag) == "yfirlit":
            yf = c
            break
    if yf is None:
        raise RuntimeError("No <yfirlit> under chosen <þing>")

    urls = {}
    for c in list(yf):
        name = strip_ns(c.tag)
        u = (c.text or "").strip()
        if u:
            urls[name] = norm_url(BASE_CURRENT, u)

    return best_n, urls


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


# -----------------------------
# Main ingestion
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="althingi.db", help="SQLite DB path")
    ap.add_argument("--reset-db", action=argparse.BooleanOptionalAction, default=True, help="Delete existing DB file before ingest (default: True)")
    ap.add_argument("--schema", default="schema_map.json", help="schema_map.json generated by check_data.py")
    ap.add_argument("--models-dir", default="app", help="Directory containing generated models.py")
    ap.add_argument("--max-records", type=int, default=None, help="Limit records per resource (debug)")
    ap.add_argument("--sleep", type=float, default=0.15, help="Sleep between requests")
    ap.add_argument("--cache-dir", default="data/cache", help="Directory for HTTP cache (default: data/cache)")
    ap.add_argument("--force-fetch", action="store_true", help="Ignore cache and re-download all resources")
    args = ap.parse_args()

    # Load schema map
    with open(args.schema, "r", encoding="utf-8") as f:
        schema_map = json.load(f)

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

    engine = create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(engine)
    if manual_models and hasattr(manual_models, "Base"):
        manual_models.Base.metadata.create_all(engine)

    fetcher = Fetcher(sleep_s=args.sleep, cache_dir=args.cache_dir, force=args.force_fetch)
    detail_cache: Dict[str, List[Tuple[int, str]]] = {}
    lthing, yfirlit = discover_current_lthing_and_yfirlit(fetcher)
    fetched_at = now_utc_iso()

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
        for r in schema_map["resources"]:
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
            session.execute(text(f'DELETE FROM "{table}" WHERE ingest_lthing=:lt AND ingest_resource=:res'),
                            {"lt": lthing, "res": name})

            # Also clear child tables
            for ct in r.get("child_tables", []):
                ctable = ct["table"]
                session.execute(text(f'DELETE FROM "{ctable}" WHERE ingest_lthing=:lt AND ingest_resource=:res'),
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

                parent = Model(**row)
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
                session.execute(text('DELETE FROM member_seat WHERE lthing=:lt'), {"lt": lthing})
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
                session.execute(text('DELETE FROM nefnd_member WHERE lthing=:lt'), {"lt": lthing})
                session.flush()
                for parent in parents:
                    nefnd_id = getattr(parent, r["attr_map"].get("id"), None) if r.get("attr_map") else None
                    url_nefndarmenn = getattr(parent, leaf_map.get("nefndarmenn"), None) if leaf_map else None
                    if not nefnd_id or not url_nefndarmenn:
                        continue
                    try:
                        xml_root = parse_xml(fetcher.get(url_nefndarmenn), url_nefndarmenn)
                        parsed = parse_nefndarmenn(xml_root, nefnd_id, lthing)
                        if parsed:
                            session.add_all([NefndMember(**p) for p in parsed])
                            session.commit()
                            print(f"[ok] nefnd {nefnd_id}: stored {len(parsed)} nefndarmenn")
                    except Exception as e:
                        print(f"[warn] failed to fetch nefndarmenn for nefnd {nefnd_id}: {e}")

            session.commit()
            inserted = len(records) - skipped_dupes
            print(f"[ok] {name}: inserted {inserted} rows into {table} ({skipped_dupes} skipped as duplicates)")
            # If we just processed þingmálalisti, collect issue documents from detail XMLs
            if name == "þingmálalisti" and manual_models and hasattr(manual_models, "IssueDocument"):
                issue_documents = []
                IssueDocModel = manual_models.IssueDocument
                # clear previous for this lthing
                session.execute(text('DELETE FROM issue_documents WHERE lthing=:lt'), {"lt": lthing})
                session.flush()
                for parent in parents:
                    detail_url = getattr(parent, r["leaf_map"].get("xml"), None) if r.get("leaf_map") else None
                    malnr = getattr(parent, r["attr_map"].get("málsnúmer"), None) if r.get("attr_map") else None
                    if not detail_url or malnr is None:
                        continue
                    try:
                        detail_xml = parse_xml(fetcher.get(detail_url), detail_url)
                        docs = parse_issue_documents(detail_xml, getattr(parent, r["attr_map"].get("málsflokkur"), None) if r.get("attr_map") else None)
                        for d in docs:
                            issue_documents.append(IssueDocModel(
                                lthing=lthing,
                                malnr=malnr,
                                malflokkur=d.get("malflokkur"),
                                skjalnr=d.get("skjalnr"),
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
                    session.commit()
                    print(f"[ok] þingmálalisti: stored {len(issue_documents)} issue_documents")
            if name == "atkvæðagreiðslur" and vote_details_to_add:
                session.add_all(vote_details_to_add)
                session.commit()
                print(f"[ok] atkvæðagreiðslur: stored {len(vote_details_to_add)} vote_details")
            if name == "þingmannalisti" and seats_to_add:
                session.add_all(seats_to_add)
                session.commit()
                print(f"[ok] þingmannalisti: stored {len(seats_to_add)} member seats")

    print(f"Done. DB: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
