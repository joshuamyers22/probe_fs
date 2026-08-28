.PHONY: sync lint test build check

sync:
	uv sync --frozen --all-extras

lint:
	uv run ruff check .

test:
	uv run pytest -q

build:
	uv build

check: lint test build
