# ThingID

ThingID is a Flask app plus data-ingestion scripts for browsing Althingi
parliament data. It downloads XML from the official sources, loads it into
SQLite via SQLAlchemy models, and serves a web UI for the current session
(issues, members, votes, committees, agenda, speeches).

## Requirements

- Python 3.10+ (works with newer versions as well)
- `pip` for dependencies

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download and build the SQLite database
make get_data

# Run the web app
make web
```

Then open `http://127.0.0.1:5000/`.

## What it does

- `scripts/check_data.py` profiles the XML endpoints and generates:
  - `app/models.py` (SQLAlchemy models)
  - `schema_map.json` (mapping used for consistent ingestion)
  - `schema_report.json` (profiling report)
- `scripts/get_data.py` downloads the XML data and populates a SQLite DB
- Flask app in `app/` reads the DB and renders HTML pages

Primary UI pages:

- `/` summary of current issues
- `/members` members list
- `/members/<id>/attendance` attendance details
- `/speeches` speeches listing
- `/votes/report` vote report
- `/committees` committees overview
- `/agenda` agenda view

## Configuration

The app reads environment variables from `.env` or `.flaskenv` (or a custom
file via `THINGID_ENV_FILE`).

Common settings:

- `THINGID_DB` or `DATABASE_URL` (default: `sqlite:///data/althingi.db`)
- `THINGID_PREFIX` or `APP_URL_PREFIX` (mount the app under a URL prefix)
- `FLASK_SECRET_KEY` or `SECRET_KEY` (Flask session signing)

## Data maintenance

Regenerate schema/profile (optional):

```bash
make check_data
```

Update the database periodically (example helper):

```bash
scripts/cron_get_data.sh
```

## Production

Use `wsgi.py` with a WSGI server (gunicorn is included in requirements):

```bash
gunicorn wsgi:application
```
