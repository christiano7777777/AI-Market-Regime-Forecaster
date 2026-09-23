"""Train the autoencoder on real data, report reconstruction loss curves and cache the
learned latent embeddings for the HMM stage.

Usage:
    python scripts/train_autoencoder.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

import config
from features.engineering import build_raw_features
from models.autoencoder import train_autoencoder


def main():
    panel = pd.read_csv(config.CACHE_DIR / "raw_panel.csv", parse_dates=["date"])
    wide = panel.pivot(index="date", columns="ticker", values="adjclose")
    wide = wide[list(config.TICKERS.keys())].ffill()
    feat = build_raw_features(wide)

    result = train_autoencoder(feat)
    print(f"train rows: {(feat.dropna().index <= result['train_end']).sum()}, "
          f"holdout rows: {(feat.dropna().index > result['train_end']).sum()}")
    print(f"final train loss: {result['train_loss_history'][-1]:.4f}")
    print(f"holdout loss:      {result['holdout_loss']:.4f}")
    print(f"(a holdout loss well above train loss would flag overfitting; "
          f"comparable or only mildly higher is the healthy case)")

    latents = result["latents"]
    print(f"\nlatents shape: {latents.shape}")
    print(latents.describe().T)

    latents.to_csv(config.CACHE_DIR / "latents.csv")
    torch.save(result["model"].state_dict(), config.CACHE_DIR / "autoencoder.pt")
    result["mean"].to_csv(config.CACHE_DIR / "scaler_mean.csv")
    result["std"].to_csv(config.CACHE_DIR / "scaler_std.csv")
    print(f"\ncached latents, model weights, and scaler -> {config.CACHE_DIR}")


if __name__ == "__main__":
    main()
