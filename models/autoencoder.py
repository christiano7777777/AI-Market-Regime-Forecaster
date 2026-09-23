"""A small autoencoder that compresses the 16-dim raw feature matrix into a low-dimensional
latent embedding, which the HMM (models/regime_hmm.py) then fits regimes on top of --
representation learning feeding a generative sequence model, rather than either alone.

Fit-on-train, freeze, apply-forward, same discipline as every other project in this
series: the scaler (mean/std) and the network's weights are both fit using ONLY rows
dated on or before `train_end`. The frozen result is then applied to the full history
(train + holdout) to produce latents -- so a holdout-period date's latent never
influenced how the encoder was trained.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def fit_scaler(train_df):
    mean = train_df.mean()
    std = train_df.std().replace(0, 1.0)
    return mean, std


def apply_scaler(df, mean, std):
    return (df - mean) / std


def train_autoencoder(feat_df, train_end=None, latent_dim=None, hidden_dim=None,
                       epochs=None, lr=None, weight_decay=None, seed=None):
    """Returns a dict: model, mean, std, latents (DataFrame covering the full history),
    train_loss_history, holdout_loss, train_end.
    """
    train_end = pd.Timestamp(train_end or config.TRAIN_END)
    latent_dim = latent_dim or config.LATENT_DIM
    hidden_dim = hidden_dim or config.HIDDEN_DIM
    epochs = epochs or config.AE_EPOCHS
    lr = lr or config.AE_LR
    weight_decay = config.AE_WEIGHT_DECAY if weight_decay is None else weight_decay
    seed = config.AE_SEED if seed is None else seed

    torch.manual_seed(seed)

    clean = feat_df.dropna()
    train_df = clean[clean.index <= train_end]
    holdout_df = clean[clean.index > train_end]
    if len(holdout_df) == 0:
        raise ValueError(f"no rows after train_end={train_end.date()} -- nothing to hold out")

    mean, std = fit_scaler(train_df)
    train_x = torch.tensor(apply_scaler(train_df, mean, std).to_numpy(), dtype=torch.float32)
    holdout_x = torch.tensor(apply_scaler(holdout_df, mean, std).to_numpy(), dtype=torch.float32)

    model = Autoencoder(train_df.shape[1], hidden_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon, _ = model(train_x)
        loss = loss_fn(recon, train_x)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    model.eval()
    with torch.no_grad():
        holdout_recon, _ = model(holdout_x)
        holdout_loss = loss_fn(holdout_recon, holdout_x).item()

        full_x = torch.tensor(apply_scaler(clean, mean, std).to_numpy(), dtype=torch.float32)
        _, full_z = model(full_x)

    latents = pd.DataFrame(full_z.numpy(), index=clean.index,
                            columns=[f"z{i}" for i in range(latent_dim)])

    return {
        "model": model, "mean": mean, "std": std, "latents": latents,
        "train_loss_history": history, "holdout_loss": holdout_loss,
        "train_end": train_end, "feature_columns": list(train_df.columns),
    }


def encode(feat_df, mean, std, model):
    """Apply an already-fitted (frozen) scaler + model to new feature rows."""
    clean = feat_df.dropna()
    x = torch.tensor(apply_scaler(clean, mean, std).to_numpy(), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        _, z = model(x)
    latent_dim = z.shape[1]
    return pd.DataFrame(z.numpy(), index=clean.index,
                         columns=[f"z{i}" for i in range(latent_dim)])
