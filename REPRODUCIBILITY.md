# Reproducibility

Use the Python version in `.python-version`, install with
`uv sync --frozen --all-extras`, and run `make check`. The committed `uv.lock`
fixes all transitive dependencies. Randomized experiments must continue to pass
explicit seeds and record their configuration alongside outputs.
