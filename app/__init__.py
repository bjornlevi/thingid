from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from flask import Flask
from flask import request
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from werkzeug.middleware.proxy_fix import ProxyFix

from .middleware import PrefixMiddleware
from .views_helper import current_lthing as _current_lthing
from .utils.sessions import load_sessions

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

    url_prefix = _normalize_url_prefix(os.environ.get("THINGID_PREFIX") or os.environ.get("APP_URL_PREFIX"))

    # Keep Flask mounted at `/`. If the app is deployed under a prefix, we rely on:
    # - PrefixMiddleware (when the proxy preserves the prefix upstream), or
    # - ProxyFix + X-Forwarded-Prefix (when the proxy strips the prefix upstream).
    app = Flask(__name__)
    app.config["URL_PREFIX"] = url_prefix

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

    # Respect reverse-proxy headers. In particular, `X-Forwarded-Prefix` lets us
    # generate correct URLs when the proxy strips the mount prefix upstream.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    app.wsgi_app = PrefixMiddleware(app.wsgi_app, url_prefix)

    @app.context_processor
    def inject_sessions():
        engine = app.config.get("ENGINE")
        selected = request.args.get("lthing", type=int)
        if not selected and engine is not None:
            try:
                with Session(engine) as session:
                    selected = _current_lthing(session)
            except Exception:
                selected = None
        sessions = load_sessions()
        session_params = {"lthing": selected} if selected else {}
        return {
            "current_lthing": selected,
            "selected_lthing": selected,
            "sessions": sessions,
            "session_params": session_params,
        }

    from . import views  # noqa: WPS433
    views.register(app)
    return app
