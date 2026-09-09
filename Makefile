.PHONY: sync lint test build check benchmark-verify benchmark-smoke benchmark-package

sync:
	uv sync --frozen --all-extras

lint:
	uv run ruff check .

test:
	uv run pytest -q

build:
	uv build

check: lint test build

benchmark-verify:
	uv run python experiments/reproduce.py verify

benchmark-smoke:
	uv run python experiments/reproduce.py smoke --output artifacts/smoke-v1

benchmark-package:
	uv run python experiments/build_benchmark.py
