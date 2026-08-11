"""
visualize.py

Generates the key figures for the report:
  1. Correlation heatmap (naive, 'before' picture)
  2. Sparse network graph from the Graphical Lasso partial correlations
  3. (Optional) regime comparison side-by-side network plots

Usage:
    python src/visualize.py
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from models import load_returns

RESULTS_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")

# TODO (Week 1): map each ticker to a sector for color-coding the network plot.
# Keep this consistent with your ticker list in data_prep.py.
SECTOR_MAP = {
    # Technology
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech",
    "NVDA": "Tech", "META": "Tech", "ADBE": "Tech", "CRM": "Tech",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "WFC": "Financials", "AXP": "Financials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    # Healthcare
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
    "MRK": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare",
    # Consumer Staples
    "PG": "Staples", "KO": "Staples", "PEP": "Staples",
    "WMT": "Staples", "COST": "Staples", "CL": "Staples",
    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "LMT": "Industrials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "AEP": "Utilities",
    # Materials
    "LIN": "Materials", "APD": "Materials", "ECL": "Materials", "FCX": "Materials",
    # Communication Services
    "DIS": "Communication", "CMCSA": "Communication", "T": "Communication", "VZ": "Communication",
    # Real Estate
    "PLD": "RealEstate", "AMT": "RealEstate", "EQIX": "RealEstate", "SPG": "RealEstate",
}

SECTOR_COLORS = {
    "Tech": "#4C72B0",
    "Financials": "#DD8452",
    "Energy": "#55A868",
    "Healthcare": "#C44E52",
    "Staples": "#8172B2",
    "Industrials": "#937860",
    "Utilities": "#DA8BC3",
    "Materials": "#8C8C8C",
    "Communication": "#CCB974",
    "RealEstate": "#64B5CD",
}


def plot_correlation_heatmap(returns: pd.DataFrame):
    corr = returns.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                xticklabels=True, yticklabels=True)
    plt.title("Naive Sample Correlation Matrix (Before Sparsity)")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "01_correlation_heatmap.png", dpi=200)
    plt.close()
    print(f"Saved {FIG_DIR / '01_correlation_heatmap.png'}")


def plot_sparse_network(partial_corr_path: Path, tickers: list, threshold: float = 1e-4):
    partial_corr = pd.read_csv(partial_corr_path, index_col=0)

    G = nx.Graph()
    G.add_nodes_from(tickers)

    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if j <= i:
                continue
            weight = partial_corr.iloc[i, j]
            if abs(weight) > threshold:
                G.add_edge(t1, t2, weight=weight)

    node_colors = [SECTOR_COLORS.get(SECTOR_MAP.get(t, ""), "#999999") for t in G.nodes]
    edge_weights = [abs(G[u][v]["weight"]) * 15 for u, v in G.edges]

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)

    plt.title("Sparse Conditional Dependency Network (Graphical Lasso)")
    plt.axis("off")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "02_sparse_network.png", dpi=200)
    plt.close()
    print(f"Saved {FIG_DIR / '02_sparse_network.png'}")
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")


def main():
    returns = load_returns()
    plot_correlation_heatmap(returns)

    partial_corr_path = RESULTS_DIR / "partial_correlations.csv"
    if partial_corr_path.exists():
        plot_sparse_network(partial_corr_path, list(returns.columns))
    else:
        print(f"{partial_corr_path} not found -- run src/models.py first.")


if __name__ == "__main__":
    main()
