import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Here we define the same sequence of layers, but store them in self.main
        self.main = nn.ModuleList([
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            # VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128, 256),
            # VAE_ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(256, 512), 
            # VAE_ResidualBlock(512, 512), 
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            # VAE_ResidualBlock(512, 512),
            # VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.3),
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        ])

        # This factor is often used in Stable Diffusion to scale latents
        # self.latent_scaling = 0.18215

    def forward(self, x):
        # x: (batch_size, 1, H, W)
        encoded_features = []
        for module in self.main:
            # The original code pads if module.stride == 2
            # We'll replicate that logic here:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            # print(f'\nX.SHAPE: {x.shape}\n')
            x = module(x)
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                encoded_features.append(x)  # Save features after downsampling

        # Now x has shape (batch_size, 8, H_enc, W_enc),
        # where 8 = 4 channels for mean + 4 channels for log_variance, 
        # but actually your code chunked them as (2, dim=1).
        # So it's actually 8 channels = 4 for mean, 4 for logvar, *if* that was the intention.
        # The code 'torch.chunk(x, 2, dim=1)' actually splits (batch_size, 8, H_enc, W_enc) into
        # two groups of 4 channels each: (batch_size, 4, H_enc, W_enc) for mean, same for logvar
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # clamp the log_variance
        log_variance = torch.clamp(log_variance, -30, 20)
        # stdev = exp(0.5 * logvar)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # sample
        noise = torch.randn_like(mean, device=x.device)
        z = mean + stdev * noise
        # z *= self.latent_scaling
        
        # print(f'\n\nZ.SHAPE= {z.shape}!!!\n\n') 
        
        # print('\nSHAPES!!!')
        # for idx, b in enumerate(encoded_features):
            # print(f'\nidx: {idx}, b.shape: {b.shape}\n')

        # Return all three for computing KL divergence
        return z, stdev, log_variance, encoded_features


