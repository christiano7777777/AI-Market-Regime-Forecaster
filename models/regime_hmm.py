"""Fits a Gaussian HMM on the autoencoder's learned latent embedding.

Same discipline as every fitted stage in this pipeline: the HMM is fit using only rows
dated on or before `train_end`; the frozen model is then applied to the full history
(train + holdout) to produce regime labels/probabilities, so a holdout-period date can
never influence the transition matrix or state distributions the model learned.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

import config


def fit_hmm(train_latents, n_states, seed=None, n_iter=None, covariance_type="full"):
    seed = config.HMM_SEED if seed is None else seed
    n_iter = n_iter or config.HMM_N_ITER
    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type,
                         n_iter=n_iter, random_state=seed)
    model.fit(train_latents.to_numpy())
    return model


def bic(model, X):
    """Lower is better. Gaussian HMM parameter count: (k-1) initial-state probs +
    k*(k-1) transition probs + k*d means + k*d*(d+1)/2 full-covariance entries."""
    k, d = model.n_components, X.shape[1]
    n_params = (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) / 2
    ll = model.score(X)
    n = X.shape[0]
    return -2 * ll + n_params * np.log(n)


def sweep_n_states(latents, train_end, n_states_list=None, seed=None, n_iter=None):
    """Fits one HMM per candidate state count (train data only), scores each on both
    train and holdout log-likelihood plus BIC. Returns a list of result dicts, richest
    info first for the caller to inspect/plot before picking a state count."""
    n_states_list = n_states_list or config.N_STATES_SWEEP
    train = latents[latents.index <= train_end]
    holdout = latents[latents.index > train_end]

    rows = []
    for k in n_states_list:
        model = fit_hmm(train, k, seed=seed, n_iter=n_iter)
        train_ll = model.score(train.to_numpy())
        holdout_ll = model.score(holdout.to_numpy())
        rows.append({
            "n_states": k,
            "train_ll_per_obs": train_ll / len(train),
            "holdout_ll_per_obs": holdout_ll / len(holdout),
            "bic": bic(model, train.to_numpy()),
            "model": model,
        })
    return rows


def implied_durations(model):
    """Expected regime duration per state, in trading days, from the transition
    matrix's diagonal (1 / (1 - p_stay))."""
    diag = np.diag(model.transmat_)
    return np.where(diag < 1, 1.0 / (1.0 - diag), np.inf)


def select_n_states(sweep_results, min_duration_days=5.0):
    """BIC alone monotonically favors more states here -- it keeps improving by carving
    out short-lived "flickering" states (a couple of days, switching back and forth)
    that behave like noise, not regimes, rather than by finding genuine new structure.
    This picks the LARGEST state count, among those tried, where every single state
    still has a plausible multi-day persistence -- the point past which BIC's further
    gains stop being trustworthy.
    """
    candidates = [r for r in sweep_results
                  if np.all(implied_durations(r["model"]) >= min_duration_days)]
    if not candidates:
        raise ValueError(f"no swept state count had every regime lasting >= "
                          f"{min_duration_days} days on average")
    return max(candidates, key=lambda r: r["n_states"])


def predict_regimes(model, latents):
    """Frozen model applied forward to the full (train + holdout) latent history.
    Returns (state_labels, state_probs) both indexed like `latents`."""
    X = latents.to_numpy()
    labels = model.predict(X)
    probs = model.predict_proba(X)
    labels_s = pd.Series(labels, index=latents.index, name="regime")
    probs_df = pd.DataFrame(probs, index=latents.index,
                             columns=[f"p_state{i}" for i in range(model.n_components)])
    return labels_s, probs_df
