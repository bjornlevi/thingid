from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from flask import Flask
from sqlalchemy import create_engine


def _load_env_file(path: Path, *, override: bool = False) -> None:
    """
    Minimal dotenv loader (so we don't require python-dotenv).

    Supports lines like:
      KEY=value
      KEY="value with spaces"
    Ignores blank lines and comments starting with '#'.
    """
    if not path.exists() or not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value


def _normalize_url_prefix(prefix: Optional[str]) -> str:
    if not prefix:
        return ""
    p = prefix.strip()
    if not p or p == "/":
        return ""
    if not p.startswith("/"):
        p = "/" + p
    return p.rstrip("/")


def create_app() -> Flask:
    """
    Minimal Flask application factory that attaches a SQLAlchemy engine and registers routes.
    """
    root_dir = Path(__file__).resolve().parent.parent
    env_file = os.environ.get("THINGID_ENV_FILE")
    if env_file:
        _load_env_file(Path(env_file), override=False)
    else:
        _load_env_file(root_dir / ".flaskenv", override=False)
        _load_env_file(root_dir / ".env", override=False)

    url_prefix = _normalize_url_prefix(
        os.environ.get("THINGID_PREFIX") or os.environ.get("APP_URL_PREFIX")
    )

    static_url_path = f"{url_prefix}/static" if url_prefix else None
    app = Flask(__name__, static_url_path=static_url_path)
    app.config["URL_PREFIX"] = url_prefix
    if url_prefix:
        app.config["APPLICATION_ROOT"] = url_prefix

    # Secret key (prefer standard Flask var, but allow project-specific name)
    app.secret_key = (
        os.environ.get("FLASK_SECRET_KEY")
        or os.environ.get("LASK_SECRET_KEY")  # common typo seen in local env files
        or os.environ.get("SECRET_KEY")
    )

    thingid_db = os.environ.get("THINGID_DB")
    if thingid_db and "://" not in thingid_db:
        # Treat as filesystem path.
        db_url = f"sqlite:///{thingid_db}"
    else:
        db_url = thingid_db
    db_url = db_url or os.environ.get("DATABASE_URL") or "sqlite:///data/althingi.db"
    engine = create_engine(db_url, future=True)
    app.config["ENGINE"] = engine

    from . import views  # noqa: WPS433
    views.register(app, url_prefix=url_prefix)
    return app
