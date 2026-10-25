# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Preclinical-to-clinical efficacy prediction: ML models (Python) and biostatistics/survival analysis (R) that predict clinical-trial efficacy from preclinical data. Python and R live side by side and exchange data through Parquet files.

## Commands

```bash
# Python setup
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Python lint + tests (same as CI)
ruff check src/python tests/python
pytest tests/python/
pytest tests/python/test_data_loader.py::test_clean_normalizes_columns_and_drops_duplicates   # single test
```

```r
# R setup (renv.lock currently lists no packages; run renv::snapshot() after adding one)
renv::restore()

# R tests (same as CI)
testthat::test_dir("tests/r")
testthat::test_file("tests/r/test_stats.R")   # single file
```

## Architecture

- **`src/python/` is not an installed package.** `tests/python/conftest.py` puts `src/python` on `sys.path`, so modules are imported flat (`from data_loader import clean`), not as `python.data_loader`. Keep new modules importable the same way.
- **R code is plain scripts, not a package.** There is no `DESCRIPTION`; R tests `source()` files using paths relative to `tests/r/` (e.g. `file.path("..", "..", "src", "r", "stats.R")`), relying on `testthat::test_dir` running each test with its directory as working directory. R functions call dependencies with `pkg::fn` (`survival::`, `arrow::`, `MASS::`) instead of `library()`.
- **Python–R interchange:** `src/python/data_loader.py` reads raw CSVs from `data/raw/` and writes cleaned data to `data/processed/*.parquet` (paths resolved from `PROJECT_ROOT`, two levels above `src/python/`); `src/r/utils.R::read_processed()` reads the same Parquet files via `arrow`.
- **Notebooks** (`notebooks/python/`, `notebooks/r/`) are exploration. Reusable code goes in `src/` with tests in `tests/`. `notebooks/r/gaussian_process.R` is a base-R port of `notebooks/python/gaussian_process.ipynb` (same sections; the scikit-learn `GaussianProcessRegressor` part is reimplemented by hand). `gaussian_process.backup.ipynb` is an older version of the notebook.

## CI (`.github/workflows/ci.yml`)

- `python` job: installs only `numpy pandas pyarrow scikit-learn pytest ruff` (not the full `requirements.txt`, which also pulls `torch`/`lightgbm`), then runs ruff and pytest. If new code imports another library, add it to that install step as well as to `requirements.txt`.
- `r` job: runs on `ubuntu-24.04` with `use-public-rspm: true`, so packages install as Posit Package Manager binaries. Building from source on the runner fails. Only `testthat` is installed; an R test that calls `survival`/`arrow` functions needs those added to the install step.
- Notebooks are neither linted nor executed in CI.

## Data and git conventions

- Everything in `data/raw/`, `data/processed/`, `data/external/` and `models/` is git-ignored (only `.gitkeep` is tracked). Never commit clinical or proprietary data or model binaries.
- Branches: `main` (stable, default) ← `develop` (integration) ← `feature/*` / `fix/*`. Open PRs from feature branches into `develop`; `develop` is merged into `main` via PR.
