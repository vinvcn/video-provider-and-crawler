-include .env

PAGES ?= 100
TERMS ?= panda,ocean sunrise,city night,forest
POOL ?= 4
KIND ?= all

.PHONY: sync test lint fmt up down migrate fetch-seed fetch-search fetch-sitemaps ingest status

sync:
	uv sync --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

up:
	docker compose up -d stock-db

down:
	docker compose down

migrate:
	uv run vpc migrate

fetch-seed:
	uv run vpc fetch-seed --pages $(PAGES)

fetch-search:
	uv run vpc fetch-search --terms "$(TERMS)" --pool $(POOL)

fetch-sitemaps:
	uv run vpc fetch-sitemaps --kind $(KIND)

ingest:
	uv run vpc ingest

status:
	uv run vpc status
