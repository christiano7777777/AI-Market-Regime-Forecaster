"""Look-ahead-bias regression test for the autoencoder: perturbing HOLDOUT-period data
must not change the TRAIN-period latents at all, since the scaler and the network's
weights are both fit using only rows on or before train_end.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from models.autoencoder import train_autoencoder


def _synthetic_features(n_days=600, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_days)
    data = rng.normal(0, 1, size=(n_days, n_features))
    return pd.DataFrame(data, index=dates, columns=[f"f{i}" for i in range(n_features)])


def test_train_period_latents_unaffected_by_holdout_perturbation():
    feat = _synthetic_features()
    train_end = feat.index[400]  # roughly two-thirds through

    kwargs = dict(train_end=train_end, latent_dim=2, hidden_dim=6, epochs=50, seed=0)
    before = train_autoencoder(feat.copy(), **kwargs)["latents"]

    perturbed = feat.copy()
    future = perturbed.index > train_end
    perturbed.loc[future] = perturbed.loc[future] * 5.0 + 10.0
    after = train_autoencoder(perturbed, **kwargs)["latents"]

    train_mask = before.index <= train_end
    pd.testing.assert_frame_equal(before.loc[train_mask], after.loc[train_mask])


def test_holdout_loss_is_finite_and_scaler_uses_train_stats_only():
    feat = _synthetic_features()
    train_end = feat.index[400]
    result = train_autoencoder(feat, train_end=train_end, latent_dim=2, hidden_dim=6,
                                epochs=50, seed=0)
    assert np.isfinite(result["holdout_loss"])

    train_df = feat.dropna()[feat.dropna().index <= train_end]
    assert np.allclose(result["mean"].to_numpy(), train_df.mean().to_numpy())
    assert np.allclose(result["std"].to_numpy(), train_df.std().replace(0, 1.0).to_numpy())


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")
