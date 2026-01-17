#!/usr/bin/env python3
"""
Load bronze CSVs into the database tables (silver step).

Usage:
  PYTHON=.venv/bin.python scripts/mint_silver.py --db data/althingi.db --models-dir app --bronze-dir data/bronze --lthing 154
  PYTHON=.venv/bin.python scripts/mint_silver.py --db data/althingi.db --models-dir app --bronze-dir data/bronze --lthing-range 150,154
  PYTHON=.venv/bin.python scripts/mint_silver.py --db data/althingi.db --models-dir app --bronze-dir data/bronze --all-lthing
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/althingi.db", help="SQLite DB path")
    ap.add_argument("--models-dir", default="app", help="Directory containing generated models.py")
    ap.add_argument("--bronze-dir", default="data/bronze", help="Base directory containing bronze CSVs")
    ap.add_argument("--lthing", type=int, default=None, help="Specific löggjafarþing number")
    ap.add_argument("--lthing-range", type=str, default=None, help="Inclusive range start,end (e.g. 150,154)")
    ap.add_argument("--all-lthing", action="store_true", help="Load all lthing folders under bronze-dir")
    args = ap.parse_args()

    bronze_dir = Path(args.bronze_dir)
    if not bronze_dir.exists():
        raise SystemExit(f"bronze directory not found: {bronze_dir}")

    if args.models_dir not in sys.path:
        sys.path.insert(0, args.models_dir)
    models = __import__("models")
    Base = getattr(models, "Base")

    model_by_table: Dict[str, Any] = {}
    for cls in Base.registry.mappers:
        mapped = cls.class_
        tab = getattr(mapped, "__tablename__", None)
        if tab:
            model_by_table[tab] = mapped

    targets: List[int] = []
    if args.all_lthing:
        targets = [int(p.name) for p in bronze_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    elif args.lthing_range:
        try:
            start_s, end_s = args.lthing_range.split(",", 1)
            start_i = int(start_s.strip())
            end_i = int(end_s.strip())
            lo, hi = sorted((start_i, end_i))
            targets = list(range(lo, hi + 1))
        except Exception as e:
            raise SystemExit(f"Invalid --lthing-range '{args.lthing_range}': {e}") from e
    elif args.lthing is not None:
        targets = [args.lthing]
    else:
        raise SystemExit("Specify --lthing, --lthing-range, or --all-lthing")

    engine = create_engine(f"sqlite:///{os.path.abspath(args.db)}", future=True, connect_args={"timeout": 60})
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        for lt in targets:
            lt_dir = bronze_dir / str(lt)
            if not lt_dir.exists():
                print(f"[warn] bronze dir missing for lthing {lt}: {lt_dir}")
                continue
            for csv_path in lt_dir.glob("*.csv"):
                table = csv_path.stem
                Model = model_by_table.get(table)
                if Model is None:
                    print(f"[warn] skipping {csv_path}: no model for table {table}")
                    continue
                rows = load_rows(csv_path)
                if not rows:
                    continue
                ingest_res = rows[0].get("ingest_resource")
                try:
                    session.execute(text(f'DELETE FROM "{table}" WHERE ingest_lthing=:lt AND ingest_resource=:res'),
                                    {"lt": lt, "res": ingest_res})
                except Exception:
                    session.rollback()
                objs = []
                for row in rows:
                    # coerce empty strings to None for cleaner inserts
                    clean = {k: (v if v != "" else None) for k, v in row.items()}
                    objs.append(Model(**clean))
                session.add_all(objs)
                session.commit()
                print(f"[ok] silver loaded {len(objs)} rows into {table} for lthing {lt}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
