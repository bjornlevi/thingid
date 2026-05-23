# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ThingID is a Flask web app for browsing Alþingi (Icelandic Parliament) data. The project has three main layers:

1. **Data pipeline** (`scripts/`): Fetches XML from official APIs, caches/transforms data, populates SQLite
2. **Database** (`app/models.py`): Auto-generated SQLAlchemy models from profiled XML schema
3. **Web app** (`app/`): Flask views, templates, and static assets serving parliament data

## Development Commands

### Setup
Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` to configure the app (see Configuration below).

### Core workflows
```bash
# Profile XML endpoints and regenerate models + schema
make check_data

# Download XML and populate SQLite database (full dataset)
make get_data

# Download specific legislative terms (parliament sessions)
make get_data thing=130          # single term
make get_data thing=129,130      # range
make get_data thing=all          # all terms

# Reset database (delete before repopulating)
make get_data reset=1

# Run Flask development server
make web
```

Then visit `http://127.0.0.1:5000/`.

### Data transformation (advanced)
```bash
# Cache XML locally before processing
make get_cache thing=130

# Transform cached XML to bronze (intermediate format)
make mint_bronze thing=130

# Transform bronze to silver (database-ready format, requires the DB)
make mint_silver thing=130
```

## Architecture Notes

**Data flow**: XML API → cache (data/cache/) → bronze (data/bronze/) → SQLite (data/althingi.db) → Flask

**Models**:
- `app/models.py`: Auto-generated from schema_map.json via check_data.py (do not edit directly)
- `app/manual_models.py`: Supplementary models and transformations added manually
- `schema_map.json`: Profile of XML structure used by get_data.py to map XML to tables

**Views** (`app/views.py`):
- Routes: `/` (issues), `/members`, `/speeches`, `/votes/report`, `/committees`, `/agenda`
- Uses SQLAlchemy Session to query the database
- Renders Jinja2 templates with vote data, member attendance, issue timelines

**Configuration**:
- Reads from `.env`, `.flaskenv`, or `THINGID_ENV_FILE`
- Key vars: `THINGID_DB` (database path), `THINGID_PREFIX` (URL mount point), `FLASK_SECRET_KEY`
- Alternate env var names: `DATABASE_URL` (instead of `THINGID_DB`), `SECRET_KEY` (instead of `FLASK_SECRET_KEY`)
- See `.env.example` for all available options
- `app/__init__.py` handles app factory, middleware (proxy headers, URL prefix), and context processors

## Key Files

| File | Purpose |
|------|---------|
| `scripts/check_data.py` | Profiles XML, generates models.py and schema_map.json |
| `scripts/get_data.py` | Downloads XML, ingests into SQLite using models |
| `scripts/get_cache.py`, `mint_bronze.py`, `mint_silver.py` | Data transformation pipeline stages |
| `app/__init__.py` | Flask app factory, middleware, configuration loading |
| `app/views.py` | URL routing and view functions (render pages with DB queries) |
| `app/views_helper.py` | Helper functions (current parliament session, sorting) |
| `app/manual_models.py` | Hand-written models and data enrichment |
| `app/utils/dates.py` | Date parsing for various Alþingi date formats |
| `app/utils/sessions.py` | Session/parliament term utilities |
| `app/middleware.py` | Custom middleware for URL prefix handling behind reverse proxies |
| `wsgi.py` | WSGI entry point for production servers (gunicorn) |

## Common Tasks

**Add a new view/page**: Edit `app/views.py` to add a route, query the database via `app/models.py`, render a template in `app/templates/`.

**Modify database schema**: Run `make check_data` to regenerate models from the XML API profiles. Note: models.py is auto-generated; add supplementary models to manual_models.py instead.

**Update data**: Run `make get_data` to re-fetch and load all terms, or `make get_data thing=N` for a specific session. For periodic updates, use `scripts/cron_get_data.sh`.

**Deploy**: Use `wsgi.py` with gunicorn: `gunicorn wsgi:application`. Configure via environment variables or `.env`.

## Debugging

- Flask server logs appear in the terminal when `make web` runs.
- SQLite database at `data/althingi.db` can be inspected with `sqlite3` CLI.
- Check `schema_report.json` for a detailed profile of all XML endpoints and fields.
