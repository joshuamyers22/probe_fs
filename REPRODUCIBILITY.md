# Reproducing the benchmark

Use this repository checkout or extract `probe-fs-reproducibility-v1.zip` into an
empty directory. The workspace bundle includes code, protocols, tables, figures
and `benchmarks/recorded-results.zip`. The Python wheel contains the library;
use the source workspace for these benchmark commands.

Use Python 3.12 from `.python-version` and install the locked dependencies:

```sh
uv sync --locked --all-extras
uv run pytest -q
```

The recorded runs used Python 3.12.14. Installation may need downloads; subsequent
simulation and recorded-evidence commands need no data download, API key or
account. Git is needed for the isolated smoke and historical-source workspaces.
Ordinary source-archive runs may record `git_commit: null`; source hashes and the
dependency lock still identify their code.

## Verify the evidence

This check requires only Python's standard library:

```sh
python experiments/reproduce.py verify
```

It verifies the archive's SHA-256 and every member's hash and size, rejects
unexpected/duplicate members, and checks paths before extraction.
`benchmarks/studies.json` lists seven recorded components. There are **2,182
checkpoints**: 22 pilot pairs, 240 expanded pairs, 240 audit records, 240 revised
pairs, 240 pairs per selection rule, and 960 context-control evaluations. Those
units differ and must not be added as independent experimental comparisons.
Configs, source snapshots, provenance, selected sets, rankings, fold results and
analysis hashes are included. Raw public-data archives, environments and execution
logs are excluded; readable figures and summaries are under `docs/`.

## Regenerate summaries without fitting

```sh
uv run python experiments/reproduce.py analyze --output artifacts/reanalysis-v1
```

Use a new directory. This extracts a verified copy, regenerates all seven study
summaries plus the paired-policy analysis, and checks their statistics against
the recorded JSON. It also regenerates plots. It never refits models or changes
original results. The audit is reaggregated from recorded post-hoc rescores.
`reanalysis-verification.json` records success.

Summary numbers, including averages of recorded timing inputs, are checked for
exact equality by default. Plot bytes can vary with platform/fonts. Linux CI
exposed confidence-interval differences of about 3e-17 from the recorded macOS
values. For cross-platform summary regeneration, opt into:

```sh
uv run python experiments/reproduce.py analyze --allow-roundoff --output artifacts/reanalysis-portable
```

This permits finite floating-point differences within relative tolerance 1e-12
or absolute tolerance 1e-14, and lists every accepted difference with its path and
both values in `reanalysis-verification.json`. Counts, types, categories and
missing values must still match exactly; evidence hashes remain exact. Larger
differences fail explicitly. The original recorded evidence is never overwritten.
These tolerances apply only to regenerated summaries, not refitted outcomes.

## Exercise fitting, analysis and resume

```sh
uv run python experiments/reproduce.py smoke --output artifacts/smoke-v1
```

This creates an isolated source workspace with no prior results or data caches.
A tiny synthetic dataset runs through both selection rules and all eight context
controls, their analyses, and a resume check for unchanged hashes and modification
times. It uses installed dependencies from the active Python environment and the
copied workspace's source. Its output is explicitly **infrastructure validation,
not scientific evidence**. Use a new output directory. CI runs this on Python 3.12.

## Rerun a historical study with its original source

Code evolved during the experiments. An old config with today's code is not
necessarily an original-method reproduction. Stage the recorded source first:

```sh
uv run python experiments/reproduce.py stage --study contexts --output artifacts/replay-contexts
cd artifacts/replay-contexts
uv sync --locked --all-extras --python 3.12.14
```

Use the recorded Python version shown in the generated `REPLAY.md`, then run its
replay command. The provenance check includes the Python patch version; another
3.12 release can run a new study but cannot resume these archived checkpoints.
Study IDs are `pilot`, `expanded`,
`audit`, `revised`, `one-se`, `argmin`, and `contexts`. Staging copies the workspace
and verified evidence, overlays the correct snapshot, and verifies every recorded
core/build-source hash. A separate local Git repository supplies the HEAD required
by older provenance code; the original repository and commits remain unchanged.
The pilot and audit share the expanded study's exact core source snapshot; this
correspondence is checked. Driver versions and configs are restored too.

Full reruns write new `results/replay-*` folders. The audit reads its archived
expanded-parent cells and remains post-hoc. The pilot replay command runs only
simulations. Its public portion additionally needs `uv run python -m pgfs.datasets`
and `--only public` in a new output directory; see the original plan and pinned
manifest for attribution and split boundaries. Financial datasets remain deferred.
Dataset licenses are unchanged.

Full grids took roughly 8–22 minutes each on the development machine. Timings
depend on hardware. Cross-platform floating-point differences may affect choices
near a threshold. Every run records its environment and rejects reuse of a result
directory if its source, config, package versions or lock changes.

## Build the review package

```sh
uv run python experiments/build_benchmark.py
```

This verifies the included evidence and writes a workspace ZIP and detached
SHA-256 under `artifacts/`. A manifest lists every included file. ZIP ordering,
permissions and timestamps are normalized: identical inputs produce identical
bytes. Maintainers with complete original results can use `--refresh-records`;
it rejects missing checkpoints or changed recorded inputs. Users need not refresh.

`make benchmark-verify`, `make benchmark-smoke`, and `make benchmark-package` are
shortcuts. `make check` retains lint/test/build checks. The source distribution
also includes experiment scripts, protocols and recorded evidence.

The Python 3.12 CI job also regenerates the recorded summaries with the explicit
roundoff option and uploads the
workspace ZIP, checksum and verification summaries as the
`probe-fs-reproducibility` workflow artifact. Each matrix job explicitly selects
its configured Python version.

See [consolidated findings](docs/BENCHMARK_RESULTS.md) and
[package validation](docs/PACKAGE_VALIDATION.md). Negative and inconclusive results
remain included; these studies do not establish broad superiority for gating.

## Completed public-data application study

The separate `benchmarks/public-application-results-v1.zip` archive contains the
180 completed application method checkpoints, 30 shared data/fold records,
historical source/configuration, runtime ledger and recorded analysis. Its
results are additional to the seven methodology components described above.

```sh
uv run python experiments/reproduce_public_results.py --output artifacts/public-results-review
```

This verifies and reanalyzes the recorded application results without fitting or
downloading data. It regenerates tables and plots, keeps counts and structure
exact, and records any finite float roundoff under the same declared tolerances.
See [application results](docs/PUBLIC_APPLICATION_RESULTS.md). The reserved Year
Prediction test partition remains unevaluated.

## Public selection-rule sensitivity

The separate `benchmarks/public-selection-results-v1.zip` contains the 120 paired
method checkpoints and their frozen protocol/source, runtime, resume and analysis
evidence. It depends on the included original public application archive for
training-only rankings, validation paths, shared folds and external references.

```sh
uv run python experiments/reproduce_public_selection.py --output artifacts/public-selection-review
```

This verifies both archives and the frozen source snapshot, extracts them into a
new directory, and regenerates four numerical reports plus plots without raw
data or model fitting. All numerical differences accepted under the declared
roundoff tolerance are logged; counts and structure remain exact. CI verifies
both archives across Python 3.10–3.12 and reanalyzes on Python 3.12.

The [full replay protocol](docs/PUBLIC_SELECTION_PLAN.md) uses public raw archives
through the same pinned loader. It refits both selection policies on each outer
fold (720 new fits), reusing original training-only ranking work. This is an
incremental experiment cost, not the standalone cost of either selector.
See [results and limitations](docs/PUBLIC_SELECTION_RESULTS.md).
