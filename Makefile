PYTHON ?= python3
OUTDIR ?= .
MODELS_DIR ?= app
DB ?= data/althingi.db
SCHEMA ?= schema_map.json
FLASK_APP ?= wsgi.py
FLASK_ENV ?= development
APP_URL_PREFIX ?=
THINGID_PREFIX ?=

.PHONY: check_data get_data web

check_data:
	$(PYTHON) scripts/check_data.py --outdir $(OUTDIR) --models-dir $(MODELS_DIR)

get_data:
	@mkdir -p $(dir $(DB))
	$(PYTHON) scripts/get_data.py --db $(DB) --schema $(SCHEMA) --models-dir $(MODELS_DIR)

web:
	FLASK_APP=$(FLASK_APP) FLASK_ENV=$(FLASK_ENV) APP_URL_PREFIX="$(APP_URL_PREFIX)" THINGID_PREFIX="$(THINGID_PREFIX)" $(PYTHON) -m flask run
