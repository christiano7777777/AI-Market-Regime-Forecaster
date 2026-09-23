"""Look-ahead-bias regression test for features/engineering.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from features.engineering import build_raw_features
import config


def _synthetic_wide_px(n_days=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    cols = {}
    for t in config.TICKERS:
        cols[t] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    return pd.DataFrame(cols, index=dates)


def test_features_unaffected_by_future_prices():
    px = _synthetic_wide_px()
    before = build_raw_features(px, vol_window=10, level_window=30)

    cutoff_idx = 250
    perturbed = px.copy()
    perturbed.iloc[cutoff_idx + 1:] *= 3.0
    after = build_raw_features(perturbed, vol_window=10, level_window=30)

    pd.testing.assert_frame_equal(before.iloc[:cutoff_idx + 1], after.iloc[:cutoff_idx + 1])


def test_no_infinite_values():
    px = _synthetic_wide_px()
    feat = build_raw_features(px, vol_window=10, level_window=30)
    assert not np.isinf(feat.to_numpy(dtype=float)).any()


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")
