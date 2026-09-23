"""Mechanical half of the dashboard refresh -- everything that needs no judgment call.

Run AFTER scripts/fetch_data.py, scripts/train_autoencoder.py, scripts/fit_hmm.py
(those three do the actual model work and must run first, in that order).

This script:
  1. Rebuilds the 16 raw features + sector-weather aggregates from the cached price
     panel and regime labels.
  2. Computes the empirical transition matrix, 5-day-ahead forecast, historical
     analogs, regime runs/transitions, and everything else the dashboard needs --
     all from cache/regime_labels.csv + cache/regime_probs.csv + cache/raw_panel.csv,
     which the three scripts above must have just refreshed.
  3. Fetches a fresh, UNSCORED batch of market/macro Guardian headlines (last 14
     days) and writes them to news_candidates.json for the agent to pick from --
     deliberately NOT run through the divergence-project's macro_score lexicon,
     since that's tuned for a different purpose (see the project's own NLP-
     robustness review for why reusing it here would misrepresent relevance).

Writes, into this same `dashboard/` folder:
  - regime_data.json      (everything about the model's own state)
  - news_candidates.json  (~40 raw headlines, unscored, for the agent to pick 5 from)

Does NOT write the final page -- picking the 5 most relevant headlines and writing
the short forecast-take paragraph needs judgment, not a formula, and is left to
whichever Claude session runs this as part of the scheduled routine.
"""
import json
import os
import re
import sys
import urllib3
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests

import config
from features.engineering import build_raw_features

CACHE = config.CACHE_DIR

# ============================================================ regime data
panel = pd.read_csv(CACHE / "raw_panel.csv", parse_dates=["date"])
wide = panel.pivot(index="date", columns="ticker", values="adjclose")
wide = wide[list(config.TICKERS.keys())].ffill()
feat = build_raw_features(wide)

labels = pd.read_csv(CACHE / "regime_labels.csv", index_col=0, parse_dates=True)["regime"]
probs = pd.read_csv(CACHE / "regime_probs.csv", index_col=0, parse_dates=True)

df = feat.join(labels, how="inner").dropna(subset=["regime"])
df["regime"] = df["regime"].astype(int)
n_states = int(labels.max()) + 1

seq = labels.reindex(feat.index).dropna().astype(int)
seq_vals = seq.values
trans_counts = np.zeros((n_states, n_states))
for i in range(len(seq_vals) - 1):
    trans_counts[seq_vals[i], seq_vals[i + 1]] += 1
transmat = trans_counts / trans_counts.sum(axis=1, keepdims=True)

runs = []
cur_s, length = seq_vals[0], 1
for v in seq_vals[1:]:
    if v == cur_s:
        length += 1
    else:
        runs.append((cur_s, length))
        cur_s, length = v, 1
runs.append((cur_s, length))
runs_df = pd.DataFrame(runs, columns=["state", "len"])
avg_dur = runs_df.groupby("state")["len"].mean()
freq_days = seq.value_counts().sort_index()

crisis_overlap = {}
for name, (s, e) in config.KNOWN_CRISIS_WINDOWS.items():
    win_idx = df.index[(df.index >= s) & (df.index <= e)]
    if len(win_idx) == 0:
        continue
    vc = df.loc[win_idx, "regime"].value_counts(normalize=True)
    crisis_overlap[name] = {int(k): float(v) for k, v in vc.to_dict().items()}

feat_z = (feat - feat.mean()) / feat.std()
sig = feat_z.join(labels, how="inner").dropna(subset=["regime"])
sig["regime"] = sig["regime"].astype(int)
signature = sig.groupby("regime").mean(numeric_only=True)

last_date = df.index.max()
current_regime = int(df.loc[last_date, "regime"])
current_probs = probs.loc[last_date].to_dict()
last_len = runs[-1][1] if runs[-1][0] == current_regime else 1

p0 = probs.loc[last_date].values.astype(float)
forecast = {}
Tk = np.eye(n_states)
for k in range(1, 6):
    Tk = Tk @ transmat
    forecast[k] = (p0 @ Tk).tolist()

SECTORS = {
    "equities": ["spy_ret", "spy_rv", "breadth_rel", "smallcap_rel", "growth_rel"],
    "vol": ["vix_level_z", "vix_chg_5d"],
    "credit": ["credit_stress"],
    "rates": ["tnx_level_z", "tnx_chg_5d", "tlt_ret", "tlt_rv"],
    "commodities": ["gld_ret", "uup_ret", "uso_ret", "uso_rv"],
}
sector_z = pd.DataFrame({name: feat_z[cols].mean(axis=1) for name, cols in SECTORS.items()})
sector_roll = sector_z.rolling(5).mean()
sector_today = {k: round(float(v), 2) for k, v in sector_roll.loc[last_date].items()}
last_week_idx = sector_roll.index[sector_roll.index <= last_date][-5:]
sector_last_week = {k: [round(float(x), 2) for x in sector_roll.loc[last_week_idx, k].values] for k in SECTORS}
regime_last_week = [int(labels.reindex(last_week_idx).iloc[i]) for i in range(len(last_week_idx))]
week_dates = [str(d.date()) for d in last_week_idx]

recent_cut = last_date - pd.Timedelta(days=45)
feat_today = feat_z.loc[last_date].values.astype(float)
feat_hist = feat_z.loc[feat_z.index < recent_cut].dropna()
fd = np.sqrt(((feat_hist.values - feat_today) ** 2).sum(axis=1))
order = np.argsort(fd)[:6]
labels_dict = labels.astype(int).to_dict()
analogs = [
    {"date": str(feat_hist.index[i].date()), "dist": float(fd[i]), "regime": labels_dict.get(feat_hist.index[i])}
    for i in order
]

recent = probs.tail(90)
spy = wide["SPY"].reindex(df.index)
chart = pd.DataFrame({"spy": spy, "regime": df["regime"]}).dropna()

run_rows = []
idx_vals = seq.index
start_i = 0
for i in range(1, len(seq_vals) + 1):
    if i == len(seq_vals) or seq_vals[i] != seq_vals[start_i]:
        run_rows.append({"state": int(seq_vals[start_i]), "start": str(idx_vals[start_i].date()),
                          "end": str(idx_vals[i - 1].date()), "days": i - start_i})
        start_i = i
transitions = run_rows[1:]

regime_out = {
    "as_of": str(last_date.date()),
    "sector_today": sector_today,
    "sector_last_week": sector_last_week,
    "week_dates": week_dates,
    "regime_last_week": regime_last_week,
    "n_states": n_states,
    "current_regime": current_regime,
    "current_probs": {k: round(float(v), 4) for k, v in current_probs.items()},
    "days_in_regime": int(last_len),
    "transmat": [[round(float(x), 4) for x in row] for row in transmat],
    "forecast": {str(k): [round(float(x), 4) for x in v] for k, v in forecast.items()},
    "avg_duration": {int(k): round(float(v), 1) for k, v in avg_dur.items()},
    "freq_days": {int(k): int(v) for k, v in freq_days.items()},
    "crisis_overlap": crisis_overlap,
    "signature": {int(k): {c: round(float(vv), 3) for c, vv in v.items()} for k, v in signature.to_dict(orient="index").items()},
    "analogs": analogs,
    "recent_probs": {str(d.date()): [round(float(x), 4) for x in row] for d, row in zip(recent.index, recent.values)},
    "chart": {str(d.date()): [round(float(p.spy), 2), int(p.regime)] for d, p in chart.iterrows()},
    "runs": run_rows,
    "transitions": transitions[-40:],
}
(HERE / "regime_data.json").write_text(json.dumps(regime_out), encoding="utf-8")
print(f"regime_data.json written -- as_of {regime_out['as_of']}, current_regime {current_regime}")

# ============================================================ raw news candidates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
GUARDIAN_API_URL = "https://content.guardianapis.com/search"
GUARDIAN_API_KEY = os.environ.get("GUARDIAN_API_KEY", "")
Q = ('("federal reserve" OR "interest rates" OR inflation OR recession OR "stock market" '
     'OR volatility OR "treasury yields" OR "credit markets" OR "bond market" OR tariff '
     'OR "market selloff" OR "market rally")')
_URL_DATE_RE = re.compile(r"/(\d{4})/([a-z]{3})/(\d{2})/")
_MONTH = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def date_from_url(url):
    m = _URL_DATE_RE.search(url or "")
    if not m:
        return ""
    y, mon, d = m.groups()
    mm = _MONTH.get(mon)
    return f"{y}-{mm:02d}-{d}" if mm else ""


try:
    if not GUARDIAN_API_KEY:
        raise RuntimeError("GUARDIAN_API_KEY environment variable is not set")
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    params = {
        "q": Q, "from-date": from_date, "to-date": to_date, "page-size": 40,
        "order-by": "newest", "show-fields": "webPublicationDate,webTitle,webUrl,sectionName",
        "api-key": GUARDIAN_API_KEY,
    }
    # verify=False: this environment has recurring "unable to get local issuer
    # certificate" failures on some outbound HTTPS hosts (see prices.py / this
    # project's own history); Guardian's search API is public and read-only.
    r = requests.get(GUARDIAN_API_URL, params=params, timeout=30, verify=False)
    r.raise_for_status()
    results = r.json()["response"]["results"]
    items = []
    for a in results:
        f = a.get("fields", {})
        url = a.get("webUrl", "")
        date = (f.get("webPublicationDate") or "")[:10] or date_from_url(url)
        title = a.get("webTitle", "")
        title = re.sub(r"\s*[–-]\s*as it happened\s*$", "", title, flags=re.I).strip()
        items.append({"date": date, "title": title, "url": url, "section": a.get("sectionName", "")})
    (HERE / "news_candidates.json").write_text(json.dumps(items), encoding="utf-8")
    print(f"news_candidates.json written -- {len(items)} articles, {from_date} -> {to_date}")
except Exception as e:
    (HERE / "news_candidates.json").write_text("[]", encoding="utf-8")
    print(f"WARNING: news fetch failed ({e!r}) -- wrote empty news_candidates.json, "
          f"the page will just show fewer/no news items this cycle")
