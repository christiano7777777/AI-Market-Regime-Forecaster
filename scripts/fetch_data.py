"""Fetch daily OHLCV for the full macro/vol/credit indicator set.

Usage:
    python scripts/fetch_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from data.prices import load_many


def main():
    panel, issues = load_many(list(config.TICKERS.keys()), tolerance_days=150, verbose=True)
    print(f"\nloaded {panel['ticker'].nunique()} of {len(config.TICKERS)} tickers, "
          f"{len(panel)} rows")
    if len(panel):
        print(f"date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    for t in config.TICKERS:
        if t not in panel["ticker"].unique():
            print(f"  MISSING: {t}")
    if issues:
        print(f"\n{len(issues)} ticker(s) flagged issues:")
        for t, msgs in issues.items():
            for m in msgs:
                print("  -", m)

    panel.to_csv(config.CACHE_DIR / "raw_panel.csv", index=False)
    print(f"\ncached -> {config.CACHE_DIR / 'raw_panel.csv'}")


if __name__ == "__main__":
    main()
