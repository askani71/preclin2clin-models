"""ML architectures (scikit-learn, LightGBM, PyTorch)."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression


def build_baseline_classifier(**kwargs) -> LogisticRegression:
    """Baseline classifier for binary clinical efficacy outcomes."""
    return LogisticRegression(max_iter=1000, **kwargs)


def build_baseline_regressor(**kwargs) -> RandomForestRegressor:
    """Baseline regressor for continuous clinical endpoints."""
    return RandomForestRegressor(random_state=42, **kwargs)
