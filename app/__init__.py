from __future__ import annotations

import os
from flask import Flask
from sqlalchemy import create_engine


def create_app() -> Flask:
    """
    Minimal Flask application factory that attaches a SQLAlchemy engine and registers routes.
    """
    app = Flask(__name__)
    db_url = os.environ.get("DATABASE_URL", "sqlite:///data/althingi.db")
    engine = create_engine(db_url, future=True)
    app.config["ENGINE"] = engine

    from . import views  # noqa: WPS433
    views.register(app)
    return app
