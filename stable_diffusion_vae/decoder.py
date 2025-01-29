
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        # *NOTES*
        ### We use group normalization because when running convolutions, two areas of a feature map
        # are more likely to be related to eachother, so we therefore want to 
        # normalize nearby batches of maps together
        # Without normalizing, each layer may be fed values that oscillate between pos/neg and low/high
        # values. Normalizing ensures each layer is fed data within a consistent range
        # Otherwise, the loss function will oscilate, and slow down training

        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (Batch_size, features, height, width)

        residue = x

        n, c, h, w = x.shape

        # Perform self attention on all pixels in image
        # (Batch_size, features, height, width) -> (batch_size, features, height * width)
        x = x.view(n, c, h * w) 

        # (batch_size, features, height * width) -> (batch_size, height * width, features)
        # Each pixel has its own embedding i.e the "feature" of the pixel

        x = x.transpose(-1, -2)
 
        x = self.attention(x)

        # (batch_size, height * width, features) -> (batch_size, height * width, features)
        x = x.transpose(-1, -2)

        # (Batch_size, features, height * width) -> (batch_size, features, height, width)
        x = x.view((n, c, h, w))

        x += residue

        return x



class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.leaky = nn.LeakyReLU(0.3)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, in_channels, height, width)

        residue = x 

        x = self.groupnorm_1(x)

        x = self.leaky(x)

        # no size change -> (Batch_size, in_channels, height, width)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = self.leaky(x)

        # no size change -> (Batch_size, in_channels, height, width)
        x = self.conv_2(x)

        # add residual so that dimensions match
        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Module):
    def __init__(self): #stg2_res, stg3_res):

        super().__init__()
            # Input remains the same
            
            
        # originally: 512 + 32
        self.stg1 = nn.Sequential(
            nn.Conv2d(512 + 32, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512), 
            VAE_AttentionBlock(512), 
            # VAE_ResidualBlock(512, 512), 
            # VAE_ResidualBlock(512, 512), 
            # VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
        )
        
        # originally: 256+512
        self.stg2 = nn.Sequential(    
            nn.Conv2d(256 + 512, 512, kernel_size=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512), 
            # VAE_ResidualBlock(512, 512), 
            # VAE_ResidualBlock(512, 512), 
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(512, 512, kernel_size=1),  # Match channel count after skip connection
        )
            
        # originally: 128 + 512
        self.stg3 = nn.Sequential(
            nn.Conv2d(128 + 512, 512, kernel_size=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            VAE_ResidualBlock(512, 256), 
            # VAE_ResidualBlock(256, 256), 
            VAE_ResidualBlock(256, 256), 
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            VAE_ResidualBlock(256, 128), 
            # VAE_ResidualBlock(128, 128), 
            VAE_ResidualBlock(128, 128), 
            nn.GroupNorm(32, 128), 
            nn.LeakyReLU(0.3), 
            nn.Conv2d(128, 64, kernel_size=3, padding=1), 
            nn.Conv2d(64, 1, kernel_size=3, padding=1), 
        )

    def forward(self, x, encoded_features):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        
        # print(f"\n\nHERE IS X.SHAPE: {x.shape}")
        
        # x /= 0.18215
        # for idx, module in enumerate(self):
            
        #     x = torch.cat([x, encoded_features.pop()], dim=1)
        #     print(f"\nIDX: {idx} IS X.SHAPE: {x.shape}")
        #     x = module(x)
        
        
        
        # if encoded_features and len(encoded_features) > 0:
        x = torch.cat([x, encoded_features.pop()], dim=1)
            # stg1_res = 512
        # print(f"\nX.SHAPE: {x.shape}\n")
        
        x = self.stg1(x)
        
        # if encoded_features and len(encoded_features) > 0:
        x = torch.cat([x, encoded_features.pop()], dim=1)
            # stg1_res = 256
        # print(f"\nX.SHAPE: {x.shape}\n")
        x = self.stg2(x)
        
        # if encoded_features and len(encoded_features) > 0:
        x = torch.cat([x, encoded_features.pop()], dim=1)
            # stg1_res = 128
        # print(f"\nX.SHAPE: {x.shape}\n")
        x = self.stg3(x)
            
        
        # x = super().forward(x)
        # print('convo!')
        # Output: (Batch_Size, 1, Height, Width)
        return x

