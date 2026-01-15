from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

import requests


def _cache_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "cache" / "loggjafarthing.xml"


def _load_xml() -> Optional[ET.Element]:
    path = _cache_path()
    if path.exists():
        try:
            return ET.fromstring(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    try:
        resp = requests.get("https://www.althingi.is/altext/xml/loggjafarthing/", timeout=15)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text, encoding="utf-8")
        return ET.fromstring(resp.text)
    except Exception:
        return None


def _tag(node: ET.Element) -> str:
    tag = node.tag
    return tag.split("}", 1)[1] if "}" in tag else tag


def load_sessions() -> List[Dict[str, Any]]:
    root = _load_xml()
    if root is None:
        return []
    sessions: List[Dict[str, Any]] = []
    for node in root.iter():
        if _tag(node) != "þing":
            continue
        num_raw = node.attrib.get("númer") or node.attrib.get("numer")
        try:
            num = int(num_raw)
        except Exception:
            continue
        period = None
        start = None
        end = None
        for child in list(node):
            ctag = _tag(child)
            if ctag == "tímabil":
                period = (child.text or "").strip() or None
            elif ctag == "þingsetning":
                start = (child.text or "").strip() or None
            elif ctag == "þinglok":
                end = (child.text or "").strip() or None
        sessions.append({
            "number": num,
            "period": period,
            "start": start,
            "end": end,
        })
    sessions.sort(key=lambda s: s["number"], reverse=True)
    return sessions
