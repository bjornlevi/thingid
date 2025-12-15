#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_data.py
Profiles Alþingi XML resources for the current session (yfirstandandi) and generates:
  - models.py        (SQLAlchemy models inferred from data)
  - schema_map.json  (mapping used by get_data.py to ingest consistently)
  - schema_report.json (full profiling report for debugging)

Usage:
  pip install requests sqlalchemy
  python check_data.py
  python check_data.py --session 157
  python check_data.py --max-records 300
  python check_data.py --outdir .
  python check_data.py --models-dir app
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import xml.etree.ElementTree as ET


BASE_CURRENT = "https://www.althingi.is/altext/xml/loggjafarthing/yfirstandandi/"


# -----------------------------
# HTTP / XML helpers
# -----------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

class Fetcher:
    def __init__(self, timeout: int = 30, sleep_s: float = 0.15):
        self.sess = requests.Session()
        self.sess.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; althingi-schema-check/1.0)"
        })
        self.timeout = timeout
        self.sleep_s = sleep_s

    def get(self, url: str, retries: int = 4) -> bytes:
        last = None
        for i in range(retries):
            try:
                r = self.sess.get(url, timeout=self.timeout)
                r.raise_for_status()
                time.sleep(self.sleep_s)
                return r.content
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

def text_or_none(e: Optional[ET.Element]) -> Optional[str]:
    if e is None:
        return None
    t = (e.text or "").strip()
    return t if t else None

def is_abs_url(u: str) -> bool:
    p = urlparse(u)
    return bool(p.scheme and p.netloc)

def norm_url(base: str, u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    return u if is_abs_url(u) else urljoin(base, u)


# -----------------------------
# Session discovery
# -----------------------------

def parse_int_maybe(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        return int(str(s).strip())
    except Exception:
        return None

def discover_sessions(root: ET.Element) -> List[ET.Element]:
    out = []
    for e in root.iter():
        if strip_ns(e.tag) == "þing" and e.attrib:
            out.append(e)
    return out

def pick_current_session(sessions: List[ET.Element]) -> ET.Element:
    best = None
    for s in sessions:
        n = parse_int_maybe(s.attrib.get("númer") or s.attrib.get("numer") or s.attrib.get("nr"))
        if n is None:
            continue
        if best is None or n > best[0]:
            best = (n, s)
    if not best:
        raise RuntimeError("Could not determine current <þing númer='...'>")
    return best[1]

def parse_yfirlit(thing_elem: ET.Element) -> Dict[str, str]:
    yf = None
    for c in list(thing_elem):
        if strip_ns(c.tag) == "yfirlit":
            yf = c
            break
    if yf is None:
        raise RuntimeError("No <yfirlit> found under <þing>")
    urls = {}
    for c in list(yf):
        name = strip_ns(c.tag)
        u = (c.text or "").strip()
        if u:
            urls[name] = u
    return urls

def parse_thingsetning(thing_elem: ET.Element) -> Optional[str]:
    for c in list(thing_elem):
        if strip_ns(c.tag) == "þingsetning":
            return text_or_none(c)
    return None


# -----------------------------
# Identifier normalization
# -----------------------------

def ascii_fold(s: str) -> str:
    s = s.replace("Þ", "Th").replace("þ", "th")
    s = s.replace("Ð", "D").replace("ð", "d")
    s = s.replace("Æ", "Ae").replace("æ", "ae")
    s = s.replace("Ö", "O").replace("ö", "o")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def sql_ident(s: str) -> str:
    s = ascii_fold(s)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "col"
    if s[0].isdigit():
        s = f"c_{s}"
    if s in {"select", "from", "where", "group", "order", "table", "index"}:
        s = f"{s}_col"
    return s

def camel(s: str) -> str:
    s = sql_ident(s)
    parts = [p for p in s.split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) or "Model"

def uniq_names(names: List[str]) -> List[str]:
    seen = {}
    out = []
    for n in names:
        base = n
        k = 1
        while n in seen:
            k += 1
            n = f"{base}_{k}"
        seen[n] = True
        out.append(n)
    return out


# -----------------------------
# Profiling / type inference
# -----------------------------

DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{2}\.\d{2}\.\d{4}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}T"),
]

def looks_like_date(s: str) -> bool:
    s = s.strip()
    return any(p.match(s) for p in DATE_PATTERNS)

def infer_scalar_type(values: List[str]) -> str:
    # Return a SQLAlchemy-ish type name: "Integer", "Float", "Text"
    if not values:
        return "Text"
    v = [str(x).strip() for x in values if x is not None and str(x).strip() != ""]
    if not v:
        return "Text"

    all_int = True
    all_num = True
    any_date = False

    for x in v:
        if looks_like_date(x):
            any_date = True
        try:
            int(x)
        except Exception:
            all_int = False
        try:
            float(x.replace(",", "."))
        except Exception:
            all_num = False

    if all_int and not any_date:
        return "Integer"
    if all_num and not any_date:
        return "Float"
    return "Text"

@dataclass
class FieldProfile:
    path: str
    count: int = 0
    max_len: int = 0
    sample_values: List[str] = None
    inferred_type: str = "Text"

    def __post_init__(self):
        if self.sample_values is None:
            self.sample_values = []

def iter_leaf_fields(record: ET.Element) -> List[Tuple[str, str]]:
    """
    Returns (path, value) for leaf elements under record.
    path is tag/tag/tag relative to record (no namespaces).
    Only includes elements with text and no element-children.
    """
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

def choose_record_path(root: ET.Element) -> Optional[str]:
    """
    Heuristic:
      - If root has repeated direct children, pick most common tag
      - Else if root has one container child whose children repeat, use "container/item"
    """
    direct = [strip_ns(c.tag) for c in list(root) if isinstance(c.tag, str)]
    if not direct:
        return None

    counts = Counter(direct)
    most, n = counts.most_common(1)[0]
    if n >= 2:
        return most

    if len(list(root)) == 1:
        cont = list(root)[0]
        kids = [strip_ns(c.tag) for c in list(cont) if isinstance(c.tag, str)]
        if kids:
            kcounts = Counter(kids)
            km, kn = kcounts.most_common(1)[0]
            if kn >= 2:
                return f"{strip_ns(cont.tag)}/{km}"

    return most

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


def guess_natural_key(attr_names: List[str], leaf_paths: List[str]) -> Optional[Tuple[str, str]]:
    """
    Returns ("attr", "id") or ("leaf", "path") for a likely stable key.
    This becomes a UNIQUE constraint, not the primary key (we always generate surrogate id).
    """
    candidates = [
        "id", "nr", "númer", "numer",
        "málsnúmer", "malnr", "málnr",
        "skjalsnúmer", "skjalnr",
        "atkvæðagreiðslunúmer", "atkvaedagreidslunumer",
        "dagbókarnúmer", "dagbokarnumer",
    ]
    lower_attrs = {a.lower(): a for a in attr_names}
    lower_leaf = {p.lower(): p for p in leaf_paths}
    for c in candidates:
        if c in lower_attrs:
            return ("attr", lower_attrs[c])
        if c in lower_leaf:
            return ("leaf", lower_leaf[c])
    return None

def build_table_name(resource_name: str, record_path: str) -> str:
    base = sql_ident(resource_name)
    rec = sql_ident(record_path.replace("/", "_"))
    return f"{base}__{rec}"


def infer_resource(resource_name: str, resource_url: str, root: ET.Element, max_records: int) -> Dict[str, Any]:
    record_path = choose_record_path(root) or "record"
    records = iter_records(root, record_path) if record_path else []
    records = records[:max_records]

    attr_values: Dict[str, List[str]] = defaultdict(list)
    leaf_profiles: Dict[str, FieldProfile] = {}
    leaf_values: Dict[str, List[str]] = defaultdict(list)
    leaf_repeat_max: Dict[str, int] = defaultdict(int)

    for r in records:
        for k, v in r.attrib.items():
            attr_values[k].append(str(v))

        leafs = iter_leaf_fields(r)
        per = Counter([p for p, _ in leafs])
        for p, cnt in per.items():
            if cnt > 1:
                leaf_repeat_max[p] = max(leaf_repeat_max[p], cnt)

        for p, v in leafs:
            leaf_values[p].append(v)
            fp = leaf_profiles.get(p)
            if fp is None:
                fp = FieldProfile(path=p)
                leaf_profiles[p] = fp
            fp.count += 1
            fp.max_len = max(fp.max_len, len(v))
            if len(fp.sample_values) < 5 and v not in fp.sample_values:
                fp.sample_values.append(v)

    for p, fp in leaf_profiles.items():
        fp.inferred_type = infer_scalar_type(leaf_values[p])

    repeated_paths = sorted([p for p in leaf_repeat_max.keys()])

    natural_key = guess_natural_key(list(attr_values.keys()), list(leaf_profiles.keys()))
    # Allow multiple vote records per mál: avoid unique constraint for atkvæðagreiðslur
    if resource_name == "atkvæðagreiðslur":
        natural_key = None

    table_name = build_table_name(resource_name, record_path)

    # Map XML attrs/leaf paths to columns
    attr_keys = list(attr_values.keys())
    scalar_leaf_paths = [p for p in leaf_profiles.keys() if p not in set(repeated_paths)]

    attr_cols = uniq_names([sql_ident(f"attr_{k}") for k in attr_keys])
    leaf_cols = uniq_names([sql_ident(f"leaf_{p.replace('/', '__')}") for p in scalar_leaf_paths])

    attr_map = dict(zip(attr_keys, attr_cols))
    leaf_map = dict(zip(scalar_leaf_paths, leaf_cols))

    # SQLA type map
    columns: List[Dict[str, Any]] = []

    # Always include ingestion metadata to make get_data trivial
    columns.extend([
        {"name": "id", "type": "Integer", "pk": True},
        {"name": "ingest_lthing", "type": "Integer", "index": True},
        {"name": "ingest_resource", "type": "Text", "index": True},
    ])

    for k in attr_keys:
        columns.append({"name": attr_map[k], "type": infer_scalar_type(attr_values[k])})
    for p in scalar_leaf_paths:
        columns.append({"name": leaf_map[p], "type": leaf_profiles[p].inferred_type})

    columns.extend([
        {"name": "source_url", "type": "Text"},
        {"name": "fetched_at", "type": "Text"},
        {"name": "raw_xml", "type": "Text"},
    ])

    # Child tables for repeated leaf paths
    child_tables = []
    for p in repeated_paths:
        child_tables.append({
            "name": f"{table_name}__{sql_ident(p.replace('/', '__'))}",
            "parent_table": table_name,
            "path": p,
            "value_type": leaf_profiles[p].inferred_type if p in leaf_profiles else "Text",
            "columns": [
                {"name": "id", "type": "Integer", "pk": True},
                {"name": "parent_id", "type": "Integer", "fk": f"{table_name}.id", "index": True},
                {"name": "seq", "type": "Integer"},
                {"name": "value", "type": leaf_profiles[p].inferred_type if p in leaf_profiles else "Text"},
                {"name": "ingest_lthing", "type": "Integer", "index": True},
                {"name": "ingest_resource", "type": "Text", "index": True},
                {"name": "source_url", "type": "Text"},
                {"name": "fetched_at", "type": "Text"},
            ]
        })

    return {
        "resource": resource_name,
        "url": resource_url,
        "root_tag": strip_ns(root.tag),
        "record_path": record_path,
        "record_count": len(records),
        "table_name": table_name,
        "natural_key": natural_key,           # ("attr","skjalsnúmer") or ("leaf","id/path") or None
        "attr_map": attr_map,                 # xml_attr_name -> column_name
        "leaf_map": leaf_map,                 # leaf_path -> column_name
        "repeated_leaf_paths": repeated_paths,
        "columns": columns,
        "child_tables": child_tables,
        "leaf_profiles": [asdict(fp) for fp in leaf_profiles.values()],
    }


# -----------------------------
# models.py generator
# -----------------------------

MODEL_HEADER = """# -*- coding: utf-8 -*-
# Auto-generated by check_data.py
from __future__ import annotations

from sqlalchemy import Column, Integer, Float, Text, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()
"""

def col_decl(c: Dict[str, Any]) -> str:
    typ = c["type"]
    args = []
    kwargs = []
    if c.get("pk"):
        kwargs.append("primary_key=True")
        kwargs.append("autoincrement=True")
    if "fk" in c:
        args.append(f'ForeignKey("{c["fk"]}")')
    if c.get("index"):
        kwargs.append("index=True")
    arg_str = ", ".join(args + kwargs)
    if arg_str:
        return f'Column({typ}, {arg_str})'
    return f'Column({typ})'

def gen_model_class(table: str, columns: List[Dict[str, Any]], natural_unique_col: Optional[str]) -> str:
    cls = camel(table)
    lines = [f"class {cls}(Base):",
             f'    __tablename__ = "{table}"']

    # Optional unique constraint on inferred natural key column
    if natural_unique_col:
        lines.append("    __table_args__ = (")
        lines.append(f'        UniqueConstraint("{natural_unique_col}", "ingest_lthing", "ingest_resource", name="uq_{sql_ident(table)}_{sql_ident(natural_unique_col)}"),')
        lines.append("    )")

    for c in columns:
        lines.append(f'    {c["name"]} = {col_decl(c)}')
    return "\n".join(lines) + "\n\n"

def generate_models_py(outdir: str, resources: List[Dict[str, Any]]) -> None:
    os.makedirs(outdir, exist_ok=True)
    parts = [MODEL_HEADER]

    for r in resources:
        # Determine the generated natural key column name (if any)
        natural_unique_col = None
        nk = r.get("natural_key")
        if nk:
            kind, key = nk
            if kind == "attr" and key in r["attr_map"]:
                natural_unique_col = r["attr_map"][key]
            elif kind == "leaf" and key in r["leaf_map"]:
                natural_unique_col = r["leaf_map"][key]

        parts.append(gen_model_class(r["table_name"], r["columns"], natural_unique_col))

        for ct in r["child_tables"]:
            parts.append(gen_model_class(ct["name"], ct["columns"], natural_unique_col=None))

    models_path = os.path.join(outdir, "models.py")
    with open(models_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    init_path = os.path.join(outdir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("")


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".", help="Output directory for schema_map.json/report (default: .)")
    ap.add_argument("--models-dir", default="app", help="Directory to write generated models.py (default: app)")
    ap.add_argument("--session", type=int, default=None, help="Override löggjafarþing number (e.g. 157)")
    ap.add_argument("--max-records", type=int, default=300, help="Max records per resource to profile (default: 300)")
    ap.add_argument("--sleep", type=float, default=0.15, help="Sleep between requests (default: 0.15s)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    fetcher = Fetcher(sleep_s=args.sleep)

    # Discover current session + yfirlit URLs
    base_xml = fetcher.get(BASE_CURRENT)
    root = parse_xml(base_xml, BASE_CURRENT)

    sessions = discover_sessions(root)
    if not sessions:
        raise RuntimeError("No <þing> elements found in yfirstandandi XML")

    if args.session is None:
        chosen = pick_current_session(sessions)
    else:
        chosen = None
        for s in sessions:
            n = parse_int_maybe(s.attrib.get("númer") or s.attrib.get("numer") or s.attrib.get("nr"))
            if n == args.session:
                chosen = s
                break
        if chosen is None:
            raise RuntimeError(f"Session {args.session} not present in yfirstandandi XML")

    lthing = parse_int_maybe(chosen.attrib.get("númer") or chosen.attrib.get("numer") or chosen.attrib.get("nr"))
    thingsetning = parse_thingsetning(chosen)
    yfirlit = parse_yfirlit(chosen)

    resources_profile: List[Dict[str, Any]] = []
    schema_map: Dict[str, Any] = {
        "generated_at": now_utc_iso(),
        "source": BASE_CURRENT,
        "lthing": lthing,
        "thingsetning": thingsetning,
        "resources": []
    }

    report: Dict[str, Any] = {
        "generated_at": schema_map["generated_at"],
        "lthing": lthing,
        "thingsetning": thingsetning,
        "yfirlit": yfirlit,
        "resources": [],
    }

    for name, url in yfirlit.items():
        url = norm_url(BASE_CURRENT, url)
        try:
            xb = fetcher.get(url)
            rroot = parse_xml(xb, url)
            prof = infer_resource(name, url, rroot, max_records=args.max_records)
            resources_profile.append(prof)

            schema_map["resources"].append({
                "name": name,
                "url": url,
                "table": prof["table_name"],
                "record_path": prof["record_path"],
                "attr_map": prof["attr_map"],
                "leaf_map": prof["leaf_map"],
                "repeated_leaf_paths": prof["repeated_leaf_paths"],
                "child_tables": [
                    {"path": ct["path"], "table": ct["name"], "value_type": ct["value_type"]}
                    for ct in prof["child_tables"]
                ],
            })

            report["resources"].append({
                "name": name,
                "url": url,
                "table": prof["table_name"],
                "record_path": prof["record_path"],
                "record_count": prof["record_count"],
                "natural_key": prof["natural_key"],
                "columns": prof["columns"],
                "child_tables": prof["child_tables"],
                "leaf_profiles": prof["leaf_profiles"],
            })

            print(f"[ok] {name}: records={prof['record_count']} table={prof['table_name']}")
        except Exception as e:
            print(f"[fail] {name}: {e}")
            report["resources"].append({"name": name, "url": url, "error": str(e)})
            schema_map["resources"].append({"name": name, "url": url, "error": str(e)})

    # Write outputs
    generate_models_py(args.models_dir, resources_profile)

    with open(os.path.join(args.outdir, "schema_map.json"), "w", encoding="utf-8") as f:
        json.dump(schema_map, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.outdir, "schema_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {os.path.join(args.models_dir, 'models.py')}")
    print(f"Wrote {os.path.join(args.outdir, 'schema_map.json')}")
    print(f"Wrote {os.path.join(args.outdir, 'schema_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
