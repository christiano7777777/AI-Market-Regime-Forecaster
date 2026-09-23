"""Central place for every tunable parameter.

This project has no P&L objective -- it's a standalone regime-detection research piece,
not a trading strategy. Validation therefore leans on statistical fit (held-out
log-likelihood), historical sanity (do regimes land on known crises), and refit
stability (do past labels stay put as more data arrives), not Sharpe.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- universe
# A deliberately diverse macro/vol/credit/rates/commodity/breadth set, not just equity
# prices -- the autoencoder's whole job is to find structure across these, so the more
# genuinely different the raw inputs are, the more interesting the learned embedding.
TICKERS = {
    "SPY": "broad equity",
    "RSP": "equal-weight equity (vs SPY = concentration/breadth proxy)",
    "IWM": "small caps (risk appetite proxy)",
    "QQQ": "growth/tech (style rotation proxy)",
    "^VIX": "implied vol",
    "HYG": "high-yield credit",
    "LQD": "investment-grade credit (vs HYG = credit stress proxy)",
    "TLT": "long treasuries (rates/duration proxy)",
    "GLD": "gold (safe-haven flows)",
    "UUP": "dollar index (USD flows)",
    "USO": "oil (commodity vol proxy)",
    "^TNX": "10-year yield level",
}
# ^IRX (13-week T-bill) was dropped: Yahoo's endpoint returns it inconsistently for this
# symbol specifically -- repeated identical requests alternated between the full ~4,900
# point history and a ~19-point stub of only the last few weeks. Confirmed reproducible
# (3+ repeat requests), not a one-off network blip, so not something a simple retry
# would reliably fix. It was only ever a secondary feature (yield curve slope alongside
# ^TNX), so it's excluded rather than built around.

# START_DATE chosen to capture the 2008 GFC while respecting HYG/UUP's 2007 inceptions --
# both launched a few months into the year, so real coverage starts ~mid-2007, still
# leaving over a year of lead-in before the crisis.
START_DATE = "2007-01-01"
END_DATE = None

REQUEST_DELAY_SEC = 0.6
MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 2.0
MIN_PRICE = 0.01
MAX_ABS_DAILY_RETURN = 0.60

# ---------------------------------------------------------------- features
REALIZED_VOL_WINDOW = 20     # trading days
LEVEL_ZSCORE_WINDOW = 252    # trading days, for VIX level / curve slope z-scoring

# ---------------------------------------------------------------- train/holdout split
# A single chronological split, not walk-forward retraining -- appropriate for a
# research model being validated for statistical soundness, not a live trading system.
TRAIN_END = "2021-12-31"

# ---------------------------------------------------------------- autoencoder
LATENT_DIM = 3
HIDDEN_DIM = 12
AE_EPOCHS = 300
AE_LR = 1e-3
AE_WEIGHT_DECAY = 1e-5
AE_SEED = 0

# ---------------------------------------------------------------- HMM
N_STATES_SWEEP = [2, 3, 4, 5]
HMM_SEED = 0
HMM_N_ITER = 200

# ---------------------------------------------------------------- validation
KNOWN_CRISIS_WINDOWS = {
    "2008 GFC": ("2008-09-01", "2009-03-31"),
    "2011 Eurozone crisis": ("2011-07-01", "2011-10-31"),
    "2015-16 China/oil selloff": ("2015-08-01", "2016-02-29"),
    "Dec 2018 selloff": ("2018-10-01", "2018-12-31"),
    "COVID crash": ("2020-02-15", "2020-04-30"),
    "2022 rate-hike bear market": ("2022-01-01", "2022-10-31"),
}
