# Sparse Statistical Learning for High-Dimensional Financial Data

**Status:** 🚧 In progress (started Aug 2026)

## Overview

This project applies sparse statistical learning methods — LASSO, Ridge, and the
Graphical Lasso — to estimate the conditional dependency structure among a set of
publicly traded stocks. Because the number of assets is often comparable to (or
larger than) the number of observations available, the naive sample covariance
matrix is unstable or singular; sparsity-inducing regularization makes reliable
estimation possible and, as a side effect, produces an interpretable "market
network" showing which assets are conditionally related once the influence of all
other assets is accounted for.

**Research question:** *(fill in once finalized — e.g., "Does the sparse
conditional dependency structure among sector-representative stocks change
meaningfully across market regimes?")*

## Motivation

*(2–4 sentences: why this matters — risk management, diversification, market
structure — and what gap in prior work this pilot study addresses. Fill in after
your literature scan in Week 1.)*

## Data

- **Source:** Yahoo Finance via `yfinance`
- **Universe:** *(e.g., N sector-representative S&P 500 stocks — list finalized in `data/raw/tickers.txt`)*
- **Period:** *(fill in date range)*
- **Frequency:** Daily adjusted close → log returns

## Method

1. Exploratory data analysis of return distributions and naive sample correlation.
2. Baseline covariance estimation (sample, Ridge-regularized).
3. Graphical Lasso for sparse precision matrix estimation, with cross-validated
   regularization strength.
4. Network construction and visualization from the estimated precision matrix.
5. Regime comparison and stability selection (bootstrap resampling).

## Repository Structure

```
sparse-financial-networks/
├── data/
│   ├── raw/            # untouched downloaded data
│   └── processed/      # cleaned, aligned log returns
├── notebooks/           # exploratory notebooks, numbered by stage
├── src/                 # reusable pipeline code
│   ├── data_prep.py
│   ├── models.py
│   ├── evaluate.py
│   └── visualize.py
├── results/
│   ├── figures/
│   └── tables/
└── report/
    ├── report.tex
    └── references.bib
```

## How to Reproduce

```bash
git clone <repo-url>
cd sparse-financial-networks
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Pipeline (once scripts are filled in):
python src/data_prep.py         # downloads + cleans data
python src/models.py            # fits baseline + Graphical Lasso models
python src/evaluate.py          # regime comparison + stability selection
python src/visualize.py         # generates figures into results/figures/
```

## Key Findings

- **Structural Regime Shift:** Comparing pre-2020 and post-2020 regimes reveals a significant reorganization in equity dependency structure, yielding a Jaccard edge similarity score of **0.405**.
- **Market Integration Post-2020:** The number of precision matrix edges grew from **195** (pre-2020) to **266** (post-2020)—a **36.4% increase** in edge density, reflecting systemic coupling during macro market shocks.
- **Methodological Robustness:** Stability selection via 20-fold bootstrap resampling identified **198 persistent edges** (appearing in $\ge 80\%$ of resamples), demonstrating that the learned network topology represents non-spurious dependencies.

## Report

The full write-up is in [`report/report.tex`](report/report.tex) (compiled PDF to
be added once finalized).

## Future Work

*(Directions for extension — e.g., time-varying graphical models, joint graphical
lasso across regimes, Bayesian sparse estimation.)*

## Author

Mahima Islam — [LinkedIn](#) · [Email](mailto:mahimabarsha999@gmail.com)
