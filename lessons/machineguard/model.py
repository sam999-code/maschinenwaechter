"""The LSTM Autoencoder - ONE canonical copy (was copy-pasted 4x).

Architecture: LSTM encoder -> 8-number latent summary ->
autoregressive LSTM decoder (feeds back its own predictions).
Anomaly signal = reconstruction MSE (see pipeline.score_windows).
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_sensors=2, hidden=16, latent=8):
        super().__init__()
        self.encoder = nn.LSTM(n_sensors, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(n_sensors, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_sensors)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h_dec = self.from_latent(self.to_latent(h[-1])).unsqueeze(0)
        inp = torch.zeros(x.shape[0], 1, x.shape[2])
        outs = []
        for t in range(x.shape[1]):
            step, (h_dec, _) = self.decoder(
                inp, (h_dec, torch.zeros_like(h_dec)))
            out_t = self.out(step)
            outs.append(out_t)
            inp = out_t  # autoregressive feedback (Lesson 6 bug fix!)
        return torch.cat(outs, dim=1)
