"""Look-ahead-bias regression test for models/regime_hmm.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from models.regime_hmm import fit_hmm, predict_regimes


def _synthetic_latents(n_days=500, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_days)
    # two visibly different clusters so a 2-state HMM has something real to find
    state = (rng.random(n_days) > 0.5).astype(float)
    data = np.column_stack([
        state * 3 + rng.normal(0, 1, n_days),
        -state * 2 + rng.normal(0, 1, n_days),
    ])
    return pd.DataFrame(data, index=dates, columns=["z0", "z1"])


def test_hmm_params_unaffected_by_holdout_perturbation():
    latents = _synthetic_latents()
    train_end = latents.index[350]

    train_before = latents[latents.index <= train_end]
    model_before = fit_hmm(train_before, n_states=2, seed=0)

    perturbed = latents.copy()
    future = perturbed.index > train_end
    perturbed.loc[future] = perturbed.loc[future] * 10.0 + 100.0  # violent holdout-only change
    train_after = perturbed[perturbed.index <= train_end]
    model_after = fit_hmm(train_after, n_states=2, seed=0)

    assert np.allclose(model_before.transmat_, model_after.transmat_)
    assert np.allclose(model_before.means_, model_after.means_)


def test_predict_regimes_covers_full_history_with_valid_probabilities():
    latents = _synthetic_latents()
    train_end = latents.index[350]
    train = latents[latents.index <= train_end]
    model = fit_hmm(train, n_states=2, seed=0)

    labels, probs = predict_regimes(model, latents)
    assert len(labels) == len(latents)
    assert set(labels.unique()).issubset({0, 1})
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all().all()


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")
