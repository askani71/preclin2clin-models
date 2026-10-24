# Preclinical-to-Clinical Efficacy Prediction

Models and statistical pipelines (Python + R) to predict clinical efficacy from preclinical data.

## 📂 Folder Layout

```text
preclin2clin-models/
├── .github/workflows/     # CI/CD pipelines (testing, linting)
├── data/
│   ├── raw/               # Immutable raw preclinical data (excluded from Git)
│   ├── processed/         # Cleaned, feature-engineered datasets (.parquet)
│   └── external/          # Public datasets, trial registries, benchmarks
├── docs/                  # Documentation, protocols, schemas
├── models/                # Serialized model artifacts (excluded from Git)
├── notebooks/
│   ├── python/            # Jupyter notebooks
│   └── r/                 # RMarkdown / Quarto
├── src/
│   ├── python/            # Python source code
│   └── r/                 # Modular R scripts
├── tests/
│   ├── python/            # pytest
│   └── r/                 # testthat
├── renv.lock              # R dependencies (renv)
└── requirements.txt       # Python dependencies
```

## 🛠️ Data Handling & Interoperability

1. **Interchange format**: datasets shared between Python and R are stored in `data/processed/` as **Apache Parquet** (`.parquet`).
2. **Data privacy**: `data/raw/` is listed in `.gitignore`. **Never commit raw clinical or proprietary preclinical data to Git.**

## ⚙️ Environment Setup

### Python
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### R
```r
install.packages("renv")
renv::restore()
```

## 🧪 Running Tests

* **Python**: `pytest tests/python/`
* **R**: `Rscript -e 'testthat::test_dir("tests/r")'`
