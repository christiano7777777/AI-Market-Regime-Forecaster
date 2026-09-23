"""Sweep HMM state count on the cached latent embeddings, pick a state count via held-out
log-likelihood and BIC, fit the final model, and cache regime labels/probabilities for
the full history.

Usage:
    python scripts/fit_hmm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from models.regime_hmm import sweep_n_states, select_n_states, implied_durations, predict_regimes


def main():
    latents = pd.read_csv(config.CACHE_DIR / "latents.csv", index_col=0, parse_dates=True)
    train_end = pd.Timestamp(config.TRAIN_END)

    results = sweep_n_states(latents, train_end, n_states_list=list(range(2, 8)))
    table = pd.DataFrame([
        {**{k: v for k, v in r.items() if k != "model"},
         "durations": [round(d, 1) for d in implied_durations(r["model"])]}
        for r in results
    ])
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")
    print(table)

    best = select_n_states(results, min_duration_days=5.0)
    print(f"\nselected n_states={best['n_states']} -- the largest state count where every "
          f"regime still has an average duration >= 5 trading days (BIC alone would have "
          f"kept preferring more states by carving out short-lived flickering states "
          f"instead of finding genuine new structure -- see durations column above)")

    model = best["model"]
    labels, probs = predict_regimes(model, latents)

    labels.to_csv(config.CACHE_DIR / "regime_labels.csv")
    probs.to_csv(config.CACHE_DIR / "regime_probs.csv")

    print(f"\nregime counts (full history):")
    print(labels.value_counts().sort_index())
    print(f"\ntransition matrix:")
    print(pd.DataFrame(model.transmat_,
                        index=[f"from {i}" for i in range(model.n_components)],
                        columns=[f"to {i}" for i in range(model.n_components)]))
    print(f"\nimplied expected regime duration (1/(1-p_stay)), trading days:")
    for i in range(model.n_components):
        p_stay = model.transmat_[i, i]
        dur = 1 / (1 - p_stay) if p_stay < 1 else float("inf")
        print(f"  state {i}: p_stay={p_stay:.3f} -> ~{dur:.1f} days")

    print(f"\ncached regime labels + probabilities -> {config.CACHE_DIR}")


if __name__ == "__main__":
    main()
