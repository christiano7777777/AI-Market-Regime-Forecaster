# Hidden Regimes — Autoencoder + HMM Regime Detection

**Objective.** Detect and characterize market regimes from a diverse macro/vol/credit
feature set. Unlike the two sibling projects, **this one has no P&L objective** — it's a
standalone research model, validated on its own statistical and economic merits, not
backtested for profit.

**Live dashboard:** see `analysis/regime_dashboard.html` (published artifact) — regime-
colored SPY chart, transition matrix, regime characterization, and crisis-alignment check.

## Architecture

1. **Raw features** (`features/engineering.py`): 16 causal features from 12 tickers —
   SPY/RSP/IWM/QQQ returns and cross-ratios (breadth, small-cap, growth rotation), VIX
   level/change, HYG-LQD credit stress, TLT/GLD/UUP/USO returns and realized vol, 10-year
   yield level/change. 2007-2026, chosen specifically to capture the 2008 GFC.
2. **Autoencoder** (`models/autoencoder.py`): a small PyTorch network compresses the
   16-dim feature vector into a 3-dim latent embedding. Scaler and weights are both fit
   on data through 2021-12-31 only, then frozen and applied forward to 2022-2026 as a
   genuine holdout — representation learning, not a hand-picked small feature set.
3. **Gaussian HMM** (`models/regime_hmm.py`): fit on the frozen latent embedding (train
   period only), producing regime labels and probabilities for the full history.

## Model selection: BIC alone was misleading

A state-count sweep (2-7) showed BIC monotonically improving through 7 states — but
inspecting each state's implied duration (`1/(1-p_stay)` from the transition matrix)
showed *why*: past 4 states, new states have 1.5-2.5 day average durations, flickering
back and forth rather than persisting like a real regime. BIC was rewarding the model for
fitting noise as extra "regimes." `select_n_states()` picks the largest state count where
every regime still averages ≥5 trading days — landing on **4 states**, not BIC's nominal
optimum.

## Validation (no Sharpe to lean on, so three different checks instead)

- **Held-out log-likelihood**: computed at every swept state count, confirms the chosen
  4-state model generalizes (holdout log-likelihood close to train, not degraded).
- **Historical alignment**: the rare, persistent "Crisis" state (387 days total) captures
  94% of the 2008 GFC and 92% of the COVID crash, but only 1-10% of the milder Dec 2018
  and 2015-16 selloffs, and the 2022 rate-hike bear market lands almost entirely in a
  *different* elevated-vol-but-still-positive state — the model distinguishes sharp panic
  crashes from grinding bear markets without ever being told the difference.
- **Refit stability**: the whole pipeline refit at 4 progressively later cutoffs
  (2015/2018/2021/2025) agrees with the final model on 81.6-83.7% of days for a shared
  2009-2014 window — moderately stable, not perfect; roughly 1 day in 6 gets relabeled as
  more data arrives, concentrated in transition/boundary periods.

## Layout

```
config.py               tickers, dates, autoencoder/HMM hyperparameters, known crisis windows
data/prices.py            direct Yahoo chart API client (same as the sibling projects)
features/engineering.py   causal raw feature matrix
models/autoencoder.py     fit-on-train/freeze/apply-forward autoencoder
models/regime_hmm.py      HMM fit, state-count selection, regime prediction
analysis/regime_dashboard.html   the published dashboard artifact
tests/                     look-ahead-bias regression tests for every fitted stage
scripts/
  fetch_data.py            data pipeline entry point
  train_autoencoder.py     fits + caches latents
  fit_hmm.py               state-count sweep + final model + regime labels
  check_alignment.py       crisis-window sanity check
  check_stability.py       multi-vintage refit-stability check
```

## Usage

```bash
pip install -r requirements.txt
python scripts/fetch_data.py
python scripts/train_autoencoder.py
python scripts/fit_hmm.py
python scripts/check_alignment.py
python scripts/check_stability.py
```
