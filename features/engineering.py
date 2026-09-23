"""Builds the raw feature matrix fed into the autoencoder.

Every feature here is causal by construction: returns are same-day realized, rolling
vol/z-score use only trailing windows. No feature at date t uses data dated after t.
"""
import numpy as np
import pandas as pd

import config


def _log_return(px):
    return np.log(px).diff()


def _realized_vol(px, window):
    return _log_return(px).rolling(window).std() * np.sqrt(252)


def _rolling_zscore(s, window):
    r = s.rolling(window)
    return (s - r.mean()) / r.std()


def build_raw_features(wide_adjclose, vol_window=None, level_window=None):
    """`wide_adjclose`: DataFrame indexed by date, columns = tickers in config.TICKERS,
    adjusted close prices. Returns a DataFrame of ~16 causal raw features, one row per
    date, meant as the autoencoder's input -- deliberately more numerous and more
    "raw" than a hand-picked small feature set, since compressing a wider, messier input
    into a learned low-dimensional embedding is the whole point of using an autoencoder
    here rather than just feeding a handful of engineered features straight into the HMM.
    """
    vol_window = vol_window or config.REALIZED_VOL_WINDOW
    level_window = level_window or config.LEVEL_ZSCORE_WINDOW
    px = wide_adjclose

    feat = pd.DataFrame(index=px.index)

    feat["spy_ret"] = _log_return(px["SPY"])
    feat["spy_rv"] = _realized_vol(px["SPY"], vol_window)

    feat["breadth_rel"] = _log_return(px["RSP"]) - _log_return(px["SPY"])
    feat["smallcap_rel"] = _log_return(px["IWM"]) - _log_return(px["SPY"])
    feat["growth_rel"] = _log_return(px["QQQ"]) - _log_return(px["SPY"])

    feat["vix_level_z"] = _rolling_zscore(px["^VIX"], level_window)
    feat["vix_chg_5d"] = px["^VIX"].diff(5)

    feat["credit_stress"] = _log_return(px["HYG"]) - _log_return(px["LQD"])

    feat["tlt_ret"] = _log_return(px["TLT"])
    feat["tlt_rv"] = _realized_vol(px["TLT"], vol_window)

    feat["gld_ret"] = _log_return(px["GLD"])
    feat["uup_ret"] = _log_return(px["UUP"])

    feat["uso_ret"] = _log_return(px["USO"])
    feat["uso_rv"] = _realized_vol(px["USO"], vol_window)

    feat["tnx_level_z"] = _rolling_zscore(px["^TNX"], level_window)
    feat["tnx_chg_5d"] = px["^TNX"].diff(5)

    return feat
