# vae_model.py

import torch
import torch.nn as nn
from encoder import VAE_Encoder
from decoder import VAE_Decoder

class VAE(nn.Module):
    """
    Combines the VAE_Encoder and VAE_Decoder into a single model.
    Returns both the reconstruction and (mu, logvar) if you want to compute KL divergence.
    """
    def __init__(self):
        super().__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()

    def forward(self, x):
        z, mu, logvar, encoded_features = self.encoder(x)
        x_recon = self.decoder(z, encoded_features)
        return x_recon, mu, logvar
