"""Direct client for Yahoo Finance's public chart API -- same approach validated in the
sibling reversal_strategy project: `yfinance`'s session/crumb handling failed outright in
this dev environment even after fixing the underlying SSL trust-store issue, while direct
calls to the same public endpoint it wraps internally work fine.
"""
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
}


def _to_unix(date_str, end=False):
    ts = pd.Timestamp(date_str) if date_str else pd.Timestamp(datetime.now(timezone.utc).date())
    if end:
        ts = ts + pd.Timedelta(days=1)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def _fetch_raw(ticker, start, end, session):
    params = {
        "period1": _to_unix(start),
        "period2": _to_unix(end, end=True),
        "interval": "1d",
        "events": "div,splits",
    }
    last_exc = None
    for attempt in range(config.MAX_RETRIES):
        try:
            r = session.get(_CHART_URL.format(ticker=ticker), params=params,
                             headers=_HEADERS, timeout=10)
            if r.status_code == 429:
                time.sleep(config.RETRY_BACKOFF_SEC * (attempt + 1))
                continue
            r.raise_for_status()
            result = r.json()["chart"]["result"]
            return result[0] if result else None
        except Exception as e:
            last_exc = e
            time.sleep(config.RETRY_BACKOFF_SEC * (attempt + 1))
    warnings.warn(f"{ticker}: failed after {config.MAX_RETRIES} attempts ({last_exc!r})")
    return None


def _parse(result, ticker):
    ts = result.get("timestamp")
    if not ts:
        return None
    quote = result["indicators"]["quote"][0]
    adj_block = result["indicators"].get("adjclose")
    adj = adj_block[0]["adjclose"] if adj_block else quote.get("close")
    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
        "open": quote.get("open"),
        "high": quote.get("high"),
        "low": quote.get("low"),
        "close": quote.get("close"),
        "adjclose": adj,
        "volume": quote.get("volume"),
    })
    df["ticker"] = ticker
    df = df.dropna(subset=["close"]).drop_duplicates(subset="date").sort_values("date")
    return df.reset_index(drop=True)


def sanity_check(df, ticker, requested_start=None):
    """Flags data-quality problems as human-readable strings; never mutates or drops rows
    itself. Whether a flagged ticker should actually be excluded is decided by the caller
    via `has_full_history`, since a bad-price flag and a missing-history flag warrant
    different responses."""
    issues = []
    price_cols = ["open", "high", "low", "close", "adjclose"]
    bad_price = (df[price_cols] < config.MIN_PRICE).any(axis=1)
    if bad_price.any():
        issues.append(f"{ticker}: {int(bad_price.sum())} rows with a price below {config.MIN_PRICE}")

    with np.errstate(invalid="ignore"):
        ret = np.log(df["adjclose"]).diff()
    extreme = ret.abs() > config.MAX_ABS_DAILY_RETURN
    if extreme.any():
        dates = df.loc[extreme, "date"].dt.date.astype(str).tolist()
        issues.append(f"{ticker}: {int(extreme.sum())} daily |log return| > "
                       f"{config.MAX_ABS_DAILY_RETURN} on {dates[:5]}")

    span_days = (df["date"].max() - df["date"].min()).days
    expected_rows = span_days * 5 / 7 * 0.90
    if len(df) < expected_rows:
        issues.append(f"{ticker}: only {len(df)} rows over a {span_days}-day span "
                       f"(expected >= {expected_rows:.0f}) -- possible gaps or late listing")

    if requested_start is not None:
        gap_days = (df["date"].min() - pd.Timestamp(requested_start)).days
        if gap_days > 30:
            issues.append(f"{ticker}: earliest row is {df['date'].min().date()}, "
                           f"{gap_days} days after the requested start {requested_start} -- "
                           f"likely a late listing or a symbol with incomplete history")
    return df, issues


def has_full_history(df, requested_start, tolerance_days=30):
    if df is None or df.empty:
        return False
    return (df["date"].min() - pd.Timestamp(requested_start)).days <= tolerance_days


def _cache_path(ticker):
    return config.CACHE_DIR / f"{ticker}.csv"


def load_ticker(ticker, start=None, end=None, refresh=False, session=None, tolerance_days=10):
    """Load one ticker's daily OHLCV, from cache if present and fresh enough, otherwise
    from Yahoo. Returns (df, issues); df is None if nothing could be loaded.

    `tolerance_days` must match the caller's own late-listing tolerance (see
    `load_many`) -- a ticker that's known to have launched partway into `start` (e.g.
    HYG/UUP in 2007) will never satisfy a tight cache-freshness check against the
    global start date, which used to force a live re-fetch on every single run
    regardless of how fresh and complete the local cache already was."""
    start = start or config.START_DATE
    end = end or config.END_DATE
    path = _cache_path(ticker)

    if not refresh and path.exists():
        cached = pd.read_csv(path, parse_dates=["date"])
        covers_start = len(cached) and cached["date"].min() <= pd.Timestamp(start) + pd.Timedelta(days=tolerance_days)
        covers_end = end is None or (len(cached) and
                                      cached["date"].max() >= pd.Timestamp(end) - pd.Timedelta(days=5))
        if covers_start and covers_end:
            return sanity_check(cached, ticker, requested_start=start)

    session = session or requests.Session()
    raw = _fetch_raw(ticker, start, end, session)
    if raw is None:
        return None, [f"{ticker}: no data returned"]
    df = _parse(raw, ticker)
    if df is None or df.empty:
        return None, [f"{ticker}: empty after parsing"]

    df.to_csv(path, index=False)
    time.sleep(config.REQUEST_DELAY_SEC)
    return sanity_check(df, ticker, requested_start=start)


def to_yahoo_symbol(ticker):
    """Yahoo Finance uses hyphens for share-class tickers (BRK-B); holdings files from
    issuers like SSGA use dots (BRK.B). Translate for the fetch only -- the caller should
    keep using the ORIGINAL ticker as the key for matching holdings weights back to price
    data, since that's what the weights are indexed by."""
    return ticker.replace(".", "-")


def load_many(tickers, start=None, end=None, refresh=False, verbose=True, tolerance_days=30):
    """Load daily OHLCV for an explicit list of tickers (ETFs, constituents, whatever).
    Each ticker is translated to Yahoo's symbol format for the fetch (see
    `to_yahoo_symbol`) but the returned panel is keyed by the ORIGINAL ticker passed in,
    so callers matching against holdings weights or other external data don't need to
    know about the translation.

    `tolerance_days` widens the late-listing exclusion window -- useful when the universe
    deliberately includes tickers known to have launched a few months into the requested
    start year (e.g. HYG/UUP in 2007), rather than being late listings that should be
    dropped.

    Returns (panel, issues): panel is a long DataFrame
    [date, ticker, open, high, low, close, adjclose, volume];
    issues is {ticker: [messages]} for anything sanity_check flagged. A ticker whose
    history doesn't reach back near `start` (a late listing, or a delisting mid-window)
    is excluded from the panel, not silently kept with a stub of rows.
    """
    req_start = start or config.START_DATE
    session = requests.Session()
    frames, all_issues = [], {}
    for i, t in enumerate(tickers):
        yahoo_t = to_yahoo_symbol(t)
        df, issues = load_ticker(yahoo_t, start=start, end=end, refresh=refresh, session=session,
                                  tolerance_days=tolerance_days)
        if issues:
            all_issues[t] = issues
        if df is not None and not df.empty and has_full_history(df, req_start, tolerance_days=tolerance_days):
            df = df.copy()
            df["ticker"] = t  # restore the caller's original ticker spelling
            frames.append(df)
        if verbose and (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(tickers)} tickers loaded")

    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return panel, all_issues
