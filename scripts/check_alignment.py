"""Sanity check: do the detected regimes actually line up with known historical crises?
Characterizes each regime statistically (mean/vol of SPY return within it) and reports
regime composition during each known crisis window from config.KNOWN_CRISIS_WINDOWS.

Usage:
    python scripts/check_alignment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config


def main():
    labels = pd.read_csv(config.CACHE_DIR / "regime_labels.csv", index_col=0, parse_dates=True)["regime"]
    panel = pd.read_csv(config.CACHE_DIR / "raw_panel.csv", parse_dates=["date"])
    spy = panel[panel["ticker"] == "SPY"].set_index("date")["adjclose"]
    spy_ret = spy.pct_change().reindex(labels.index)

    print("regime characterization (SPY daily return stats while in each regime):")
    char = pd.DataFrame({
        "n_days": labels.value_counts().sort_index(),
        "spy_mean_ret": spy_ret.groupby(labels).mean(),
        "spy_ann_vol": spy_ret.groupby(labels).std() * (252 ** 0.5),
    })
    char["spy_ann_ret"] = char["spy_mean_ret"] * 252
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")
    print(char[["n_days", "spy_ann_ret", "spy_ann_vol"]])

    print("\nregime composition during known crisis windows:")
    for name, (start, end) in config.KNOWN_CRISIS_WINDOWS.items():
        window = labels.loc[start:end]
        if len(window) == 0:
            print(f"  {name}: no data in this window")
            continue
        composition = window.value_counts(normalize=True).sort_index()
        comp_str = ", ".join(f"state{int(k)}={v:.0%}" for k, v in composition.items())
        print(f"  {name} ({start} to {end}, {len(window)} days): {comp_str}")


if __name__ == "__main__":
    main()
