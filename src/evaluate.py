"""
evaluate.py

Week 3 core analysis:
1. Regime comparison -- fit Graphical Lasso separately on two (or more) time
   windows and quantify how the network structure differs.
2. Stability selection -- bootstrap resample the data, refit, and check which
   edges appear consistently. This is what shows methodological rigor rather
   than a single point-estimate network.

Usage:
    python src/evaluate.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import resample

from models import load_returns, fit_graphical_lasso, precision_to_partial_corr

RESULTS_DIR = Path("results/tables")

# TODO (Week 3, Step 9): finalize your regime split. Options:
#   - Fixed calendar split, e.g. pre-2020 vs 2020-onward
#   - Volatility-based split, e.g. top/bottom tercile of realized VIX
REGIME_A_RANGE = ("2015-01-01", "2019-12-31")
REGIME_B_RANGE = ("2020-01-01", "2024-12-31")


def edge_set(partial_corr: np.ndarray, threshold: float = 1e-4) -> set:
    """Return the set of (i, j) index pairs with a nonzero partial correlation."""
    idx = np.argwhere(np.abs(np.triu(partial_corr, k=1)) > threshold)
    return set(map(tuple, idx))


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compare_regimes(returns: pd.DataFrame):
    regime_a = returns.loc[REGIME_A_RANGE[0]:REGIME_A_RANGE[1]]
    regime_b = returns.loc[REGIME_B_RANGE[0]:REGIME_B_RANGE[1]]

    print(f"Regime A: {regime_a.shape[0]} obs | Regime B: {regime_b.shape[0]} obs")

    model_a = fit_graphical_lasso(regime_a)
    model_b = fit_graphical_lasso(regime_b)

    pc_a = precision_to_partial_corr(model_a.precision_)
    pc_b = precision_to_partial_corr(model_b.precision_)

    edges_a = edge_set(pc_a)
    edges_b = edge_set(pc_b)

    similarity = jaccard_similarity(edges_a, edges_b)

    print(f"Regime A edges: {len(edges_a)} | Regime B edges: {len(edges_b)}")
    print(f"Jaccard similarity of edge sets: {similarity:.3f}")

    # TODO: also compare average degree, density, and identify which specific
    # edges appeared/disappeared -- this is where your actual "finding" comes from.

    return {
        "regime_a_edges": len(edges_a),
        "regime_b_edges": len(edges_b),
        "jaccard_similarity": similarity,
    }


def stability_selection(returns: pd.DataFrame, n_bootstrap: int = 100, threshold: float = 0.8):
    """
    Refit Graphical Lasso on bootstrap resamples of the data and record how
    often each edge appears. Edges appearing in >= threshold fraction of
    resamples are considered 'stable'.

    NOTE: this is computationally heavier -- consider reducing n_bootstrap
    during development and increasing it for the final run.
    """
    n_assets = returns.shape[1]
    edge_counts = np.zeros((n_assets, n_assets))

    for b in range(n_bootstrap):
        sample = resample(returns, replace=True, random_state=b)
        try:
            model = fit_graphical_lasso(sample, cv=3)  # fewer CV folds for speed
            pc = precision_to_partial_corr(model.precision_)
            mask = np.abs(pc) > 1e-4
            edge_counts += mask
        except Exception as e:
            print(f"Bootstrap iteration {b} failed: {e}")
            continue

        if (b + 1) % 10 == 0:
            print(f"  Bootstrap {b + 1}/{n_bootstrap} done")

    edge_freq = edge_counts / n_bootstrap
    stable_edges = edge_freq >= threshold

    print(f"Stable edges (appearing in >= {threshold:.0%} of resamples): "
          f"{int(np.triu(stable_edges, k=1).sum())}")

    return edge_freq


def main():
    returns = load_returns()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Regime Comparison ===")
    regime_results = compare_regimes(returns)
    pd.Series(regime_results).to_csv(RESULTS_DIR / "regime_comparison.csv")

    print("\n=== Stability Selection ===")
    # Reduce n_bootstrap to ~20 while developing/debugging, raise to 100+ for final run
    edge_freq = stability_selection(returns, n_bootstrap=20)
    pd.DataFrame(edge_freq, index=returns.columns, columns=returns.columns) \
        .to_csv(RESULTS_DIR / "edge_stability_frequencies.csv")


if __name__ == "__main__":
    main()
