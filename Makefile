PYTHON ?= python3
OUTDIR ?= .
MODELS_DIR ?= app
DB ?= data/althingi.db
SCHEMA ?= schema_map.json
thing ?=
FLASK_APP ?= wsgi.py
FLASK_ENV ?= development
APP_URL_PREFIX ?=
THINGID_PREFIX ?=

.PHONY: check_data get_data web

check_data:
	$(PYTHON) scripts/check_data.py --outdir $(OUTDIR) --models-dir $(MODELS_DIR)

get_data:
	@mkdir -p $(dir $(DB))
	@reset_flag=$$(test "$(reset)" = "1" && echo "--reset-db" || echo "--no-reset-db"); \
	@if echo "$(thing)" | grep -q ',' ; then \
		$(PYTHON) scripts/get_data.py --db $(DB) --schema $(SCHEMA) --models-dir $(MODELS_DIR) $$reset_flag --lthing-range $(thing); \
	elif [ "$(thing)" = "all" ]; then \
		$(PYTHON) scripts/get_data.py --db $(DB) --schema $(SCHEMA) --models-dir $(MODELS_DIR) $$reset_flag --all-lthing; \
	elif [ -n "$(thing)" ]; then \
		$(PYTHON) scripts/get_data.py --db $(DB) --schema $(SCHEMA) --models-dir $(MODELS_DIR) $$reset_flag --lthing $(thing); \
	else \
		$(PYTHON) scripts/get_data.py --db $(DB) --schema $(SCHEMA) --models-dir $(MODELS_DIR) $$reset_flag; \
	fi

get_cache:
	@mkdir -p data/cache
	@if echo "$(thing)" | grep -q ',' ; then \
		$(PYTHON) scripts/get_cache.py --schema $(SCHEMA) --cache-dir data/cache --lthing-range $(thing); \
	elif [ "$(thing)" = "all" ]; then \
		$(PYTHON) scripts/get_cache.py --schema $(SCHEMA) --cache-dir data/cache --all-lthing; \
	elif [ -n "$(thing)" ]; then \
		$(PYTHON) scripts/get_cache.py --schema $(SCHEMA) --cache-dir data/cache --lthing $(thing); \
	else \
		$(PYTHON) scripts/get_cache.py --schema $(SCHEMA) --cache-dir data/cache; \
	fi

mint_bronze:
	@mkdir -p data/bronze
	@if echo "$(thing)" | grep -q ',' ; then \
		$(PYTHON) scripts/mint_bronze.py --schema $(SCHEMA) --cache-dir data/cache --outdir data/bronze --lthing-range $(thing); \
	elif [ "$(thing)" = "all" ]; then \
		$(PYTHON) scripts/mint_bronze.py --schema $(SCHEMA) --cache-dir data/cache --outdir data/bronze --all-lthing; \
	elif [ -n "$(thing)" ]; then \
		$(PYTHON) scripts/mint_bronze.py --schema $(SCHEMA) --cache-dir data/cache --outdir data/bronze --lthing $(thing); \
	else \
		$(PYTHON) scripts/mint_bronze.py --schema $(SCHEMA) --cache-dir data/cache --outdir data/bronze; \
	fi

mint_silver:
	@if [ -z "$(thing)" ]; then \
		echo "Specify thing=<n>, thing=a,b, or thing=all"; exit 1; \
	elif [ "$(thing)" = "all" ]; then \
		$(PYTHON) scripts/mint_silver.py --db $(DB) --models-dir $(MODELS_DIR) --bronze-dir data/bronze --all-lthing; \
	elif echo "$(thing)" | grep -q ',' ; then \
		$(PYTHON) scripts/mint_silver.py --db $(DB) --models-dir $(MODELS_DIR) --bronze-dir data/bronze --lthing-range $(thing); \
	else \
		$(PYTHON) scripts/mint_silver.py --db $(DB) --models-dir $(MODELS_DIR) --bronze-dir data/bronze --lthing $(thing); \
	fi

web:
	FLASK_APP=$(FLASK_APP) FLASK_ENV=$(FLASK_ENV) APP_URL_PREFIX="$(APP_URL_PREFIX)" THINGID_PREFIX="$(THINGID_PREFIX)" $(PYTHON) -m flask run
