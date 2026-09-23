"""Refit-stability check: retrain the whole pipeline (autoencoder + HMM) at several
progressively later `train_end` cutoffs, and check whether regime labels for a COMMON
historical period stay consistent as more data becomes available -- the same discipline
as the walk-forward checks in the other two projects, applied to label consistency
rather than P&L. A regime model that keeps relabeling 2010 every time it sees a new
year of data isn't trustworthy, however good its held-out likelihood looks.

HMM state indices are arbitrary between independent fits (state 2 in one fit and state 0
in another can be the same regime), so states are matched across vintages by nearest
centroid on their (mean SPY return, vol) characterization over the shared overlap window
before comparing labels directly.

Usage:
    python scripts/check_stability.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import config
from features.engineering import build_raw_features
from models.autoencoder import train_autoencoder
from models.regime_hmm import sweep_n_states, select_n_states, predict_regimes

VINTAGES = ["2015-12-31", "2018-12-31", "2021-12-31", "2025-12-31"]
OVERLAP_START = "2009-01-01"
OVERLAP_END = "2014-12-31"


def characterize(labels, spy_ret):
    return pd.DataFrame({
        "mean_ret": spy_ret.groupby(labels).mean(),
        "vol": spy_ret.groupby(labels).std(),
    })


def match_states(ref_char, other_char):
    ref = ref_char[["mean_ret", "vol"]].to_numpy()
    other = other_char[["mean_ret", "vol"]].to_numpy()
    cost = np.linalg.norm(ref[:, None, :] - other[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return dict(zip(other_char.index[col_ind], ref_char.index[row_ind]))


def main():
    panel = pd.read_csv(config.CACHE_DIR / "raw_panel.csv", parse_dates=["date"])
    wide = panel.pivot(index="date", columns="ticker", values="adjclose")
    wide = wide[list(config.TICKERS.keys())].ffill()
    feat = build_raw_features(wide)
    spy_ret = wide["SPY"].pct_change()

    label_sets = {}
    for vintage in VINTAGES:
        ae = train_autoencoder(feat, train_end=vintage)
        results = sweep_n_states(ae["latents"], pd.Timestamp(vintage),
                                  n_states_list=list(range(2, 7)))
        best = select_n_states(results, min_duration_days=5.0)
        labels, _ = predict_regimes(best["model"], ae["latents"])
        label_sets[vintage] = labels
        print(f"vintage {vintage}: selected n_states={best['n_states']}")

    overlap_idx = None
    for labels in label_sets.values():
        idx = labels.loc[OVERLAP_START:OVERLAP_END].index
        overlap_idx = idx if overlap_idx is None else overlap_idx.intersection(idx)

    ref_vintage = VINTAGES[-1]
    ref_labels = label_sets[ref_vintage].reindex(overlap_idx)
    ref_char = characterize(ref_labels, spy_ret.reindex(overlap_idx))

    print(f"\noverlap window: {overlap_idx.min().date()} -> {overlap_idx.max().date()}, "
          f"{len(overlap_idx)} days\n")
    for vintage in VINTAGES:
        labels = label_sets[vintage].reindex(overlap_idx)
        char = characterize(labels, spy_ret.reindex(overlap_idx))
        mapping = match_states(ref_char, char)
        matched = labels.map(mapping)
        agree = (matched == ref_labels).mean()
        print(f"vintage train_end={vintage}: agreement with final vintage "
              f"({ref_vintage}) over the overlap window = {agree:.1%}")


if __name__ == "__main__":
    main()
