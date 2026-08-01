"""
data_prep.py

Downloads adjusted close prices for a chosen set of tickers, computes log
returns, and saves both raw and processed data.

Usage:
    python src/data_prep.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# TODO (Week 1, Day 6-7): finalize your ticker universe.
# Aim for ~50-100 liquid, sector-diverse stocks. Keep the list in a plain text
# file (data/raw/tickers.txt) so the choice is documented and reproducible.
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",   # Tech
    "JPM", "BAC", "GS", "MS", "C",             # Financials
    "XOM", "CVX", "COP",                        # Energy
    "JNJ", "PFE", "UNH", "MRK",                 # Healthcare
    "PG", "KO", "PEP", "WMT",                   # Consumer staples
    # TODO: add more to reach your target count, and confirm sector labels
]

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"


def download_prices(tickers, start, end) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    data.to_csv(RAW_DIR / "adj_close_prices.csv")
    return data


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert price series to log returns and drop rows with any missing data."""
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    # TODO (Week 2): decide on your missing-data policy explicitly, and document
    # it in the README. Dropping any row with *any* NaN is conservative; you may
    # instead want to drop tickers with too much missing history first.
    log_returns = log_returns.dropna(axis=0, how="any")
    return log_returns


def save_processed(log_returns: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log_returns.to_csv(PROCESSED_DIR / "log_returns.csv")
    print(f"Saved processed log returns: {log_returns.shape[0]} rows, "
          f"{log_returns.shape[1]} tickers -> {PROCESSED_DIR / 'log_returns.csv'}")


def main():
    print(f"Downloading {len(TICKERS)} tickers from {START_DATE} to {END_DATE}...")
    prices = download_prices(TICKERS, START_DATE, END_DATE)
    log_returns = compute_log_returns(prices)
    save_processed(log_returns)


if __name__ == "__main__":
    main()
