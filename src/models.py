"""
models.py

Wrappers around sample covariance, Ridge-regularized covariance, and the
Graphical Lasso for sparse precision matrix estimation.

Usage:
    python src/models.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.covariance import GraphicalLassoCV, LedoitWolf, empirical_covariance

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")


def load_returns() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "log_returns.csv", index_col=0, parse_dates=True)


def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Naive sample covariance -- the 'before' baseline. Often noisy/unstable
    when the number of assets is large relative to the number of observations."""
    return empirical_covariance(returns.values)


def ridge_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Ridge-regularized (Ledoit-Wolf shrinkage) covariance -- an intermediate
    baseline between the noisy sample covariance and the sparse Graphical Lasso."""
    lw = LedoitWolf().fit(returns.values)
    return lw.covariance_


def fit_graphical_lasso(returns: pd.DataFrame, alphas=4, cv=5):
    """
    Fit Graphical Lasso with cross-validated regularization strength.

    TODO (Week 2, Step 7): consider replacing GraphicalLassoCV's default alpha
    grid with a manually specified grid -- this is more defensible in a paper
    than relying on library defaults, and lets you show the CV curve.

    Returns
    -------
    model : fitted GraphicalLassoCV instance
        model.precision_ is the estimated sparse precision matrix
        model.covariance_ is the estimated covariance matrix
        model.alpha_ is the selected regularization strength
    """
    model = GraphicalLassoCV(alphas=alphas, cv=cv, max_iter=1000)
    model.fit(returns.values)
    return model


def precision_to_partial_corr(precision: np.ndarray) -> np.ndarray:
    """Convert a precision matrix to partial correlations for interpretability."""
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)
    return partial_corr


def main():
    returns = load_returns()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Fitting sample covariance (baseline 1)...")
    cov_sample = sample_covariance(returns)

    print("Fitting Ledoit-Wolf / Ridge covariance (baseline 2)...")
    cov_ridge = ridge_covariance(returns)

    print("Fitting Graphical Lasso (core method)...")
    glasso = fit_graphical_lasso(returns)
    print(f"Selected alpha (regularization strength): {glasso.alpha_:.5f}")

    partial_corr = precision_to_partial_corr(glasso.precision_)

    n_edges = int((np.abs(np.triu(partial_corr, k=1)) > 1e-4).sum())
    print(f"Number of nonzero edges in the sparse network: {n_edges}")

    # Save outputs for downstream evaluation/visualization steps
    pd.DataFrame(glasso.precision_, index=returns.columns, columns=returns.columns) \
        .to_csv(RESULTS_DIR / "precision_matrix.csv")
    pd.DataFrame(partial_corr, index=returns.columns, columns=returns.columns) \
        .to_csv(RESULTS_DIR / "partial_correlations.csv")


if __name__ == "__main__":
    main()
