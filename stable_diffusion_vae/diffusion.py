# import torch
# from torch import nn
# from torch.nn import functional as F
# from attention import SelfAttention  # CrossAttention is no longer used.

# class TimeEmbedding(nn.Module):
#     def __init__(self, n_embd):
#         super().__init__()
#         self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
#         self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)
#         self.leaky = nn.LeakyReLU(0.3)

#     def forward(self, x):
#         # x: (1, 320)  -> for example

#         x = self.linear_1(x)
#         x = self.leaky(x)
#         x = self.linear_2(x)
#         return x


# class UNET_ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, n_time=1280):
#         super().__init__()
#         # GN1
#         self.groupnorm_feature = nn.GroupNorm(4, in_channels)
#         self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.linear_time = nn.Linear(n_time, out_channels)
#         self.leaky = nn.LeakyReLU(0.3)

#         #GN 2
#         self.groupnorm_merged = nn.GroupNorm(4, out_channels)
#         self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

#         if in_channels == out_channels:
#             self.residual_layer = nn.Identity()
#         else:
#             self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
#     def forward(self, feature, time):
#         # feature: (Batch_Size, In_Channels, Height, Width)
#         # time: (1, 1280)

#         residue = feature
        
#         #GN 1
#         # feature = self.groupnorm_feature(feature)
#         feature = self.leaky(feature)
#         feature = self.conv_feature(feature)
        
#         time = self.leaky(time)
#         time = self.linear_time(time)
        
#         # Broadcast time over spatial dimensions
#         merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
#         # GN2
#         # merged = self.groupnorm_merged(merged)
#         merged = self.leaky(merged)
#         merged = self.conv_merged(merged)
        
#         return merged + self.residual_layer(residue)


# class UNET_AttentionBlock(nn.Module):

#     def __init__(self, n_head: int, n_embd: int):
#         super().__init__()
#         channels = n_head * n_embd
        
#         # GN 3
#         self.groupnorm = nn.GroupNorm(4, channels, eps=1e-6)
#         self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

#         # Self-attention
#         self.layernorm_1 = nn.LayerNorm(channels)
#         self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)

#         # Feed-forward (GeGLU)
#         self.layernorm_2 = nn.LayerNorm(channels)
#         self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
#         self.linear_geglu_2 = nn.Linear(4 * channels, channels)

#         self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
#     def forward(self, x):
#         # x: (Batch_Size, Features, Height, Width)
#         residue_long = x

#         # GN 3
#         # x = self.groupnorm(x)
#         x = self.conv_input(x)
        
#         n, c, h, w = x.shape
        
#         # Flatten the spatial dimensions
#         x = x.view((n, c, h * w))
#         x = x.transpose(-1, -2)  # (n, h*w, c)

#         # ----- Self-Attention -----
#         residue_short = x
#         x = self.layernorm_1(x)
#         x = self.attention_1(x)
#         x = x + residue_short

#         # ----- Feed-Forward (GeGLU) -----
#         residue_short = x
#         x = self.layernorm_2(x)

#         # GeGLU
#         x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
#         x = x * F.gelu(gate)
#         x = self.linear_geglu_2(x)
#         x = x + residue_short

#         # Reshape back to (Batch_Size, Features, Height, Width)
#         x = x.transpose(-1, -2)
#         x = x.view((n, c, h, w))

#         return self.conv_output(x) + residue_long


# class Upsample(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=2, mode='nearest')
#         return self.conv(x)


# class SwitchSequential(nn.Sequential):
#     """
#     Allows passing the `time` embedding to ResidualBlocks
#     while removing context usage entirely.
#     """
#     def forward(self, x, time):
#         for layer in self:
#             if isinstance(layer, UNET_AttentionBlock):
#                 x = layer(x)
#             elif isinstance(layer, UNET_ResidualBlock):
#                 x = layer(x, time)
#             else:
#                 x = layer(x)
#         return x


# class UNET(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoders = nn.ModuleList([
#             # (Batch_Size, 32, Height/8, Width/8) -> (Batch_Size, 320, Height/8, Width/8)
#             SwitchSequential(nn.Conv2d(32, 320, kernel_size=3, padding=1)),
            
#             # (Batch_Size, 320) -> ResBlock -> SelfAttention
#             SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
#             SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
#             # Downsample
#             SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
#             # (320 -> 640) -> ResBlock -> SelfAttention
#             SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
#             SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
#             # Downsample
#             SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
#             # (640 -> 1280) -> ResBlock -> SelfAttention
#             SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
#             SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
#             # Downsample
#             SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
#             # Deeper ResBlock chain
#             SwitchSequential(UNET_ResidualBlock(1280, 1280)),
#             SwitchSequential(UNET_ResidualBlock(1280, 1280)),
#         ])

#         self.bottleneck = SwitchSequential(
#             UNET_ResidualBlock(1280, 1280),
#             UNET_AttentionBlock(8, 160),
#             UNET_ResidualBlock(1280, 1280),
#         )
        
#         self.decoders = nn.ModuleList([
#             SwitchSequential(UNET_ResidualBlock(2560, 1280)),
#             SwitchSequential(UNET_ResidualBlock(2560, 1280)),
#             SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
#             SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
#             SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
#             SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
#             SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
#             SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
#             SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
#             SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
#             SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
#             SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
#         ])

#     def forward(self, x, time):
#         # x: (Batch_Size, 32, Height/8, Width/8)
#         # time: (1, 1280)

#         skip_connections = []
#         for layers in self.encoders:
#             x = layers(x, time)
#             skip_connections.append(x)

#         x = self.bottleneck(x, time)

#         for layers in self.decoders:
#             x = torch.cat((x, skip_connections.pop()), dim=1)
#             x = layers(x, time)
        
#         return x


# class UNET_OutputLayer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         # GN 4
#         self.groupnorm = nn.GroupNorm(4, in_channels)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.leaky = nn.LeakyReLU(0.3)

#     def forward(self, x):
#         # GN 4
#         # x = self.groupnorm(x)
#         x = self.leaky(x)
#         x = self.conv(x)
#         return x


# class Diffusion(nn.Module):
#     """
#     Diffusion model without cross-attention/context.
#     """
#     def __init__(self):
#         super().__init__()
#         self.time_embedding = TimeEmbedding(320)
#         self.unet = UNET()
#         self.final = UNET_OutputLayer(320, 32)
    
#     def forward(self, latent, time):
#         """
#         :param latent: (Batch_Size, 4=32, Height/8, Width/8)
#         :param time:   (1, 320)
#         """
#         time = self.time_embedding(time)
#         output = self.unet(latent, time)
#         output = self.final(output)
#         return output

import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention  # CrossAttention is no longer used.

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # self.linear_1 = nn.Linear(n_embd, 4 * n_embd)  # Old
        self.linear_1 = nn.Linear(n_embd, 2 * n_embd)
        # self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)  # Old
        self.linear_2 = nn.Linear(2 * n_embd, 2 * n_embd)
        self.leaky = nn.LeakyReLU(0.3)  # COMMENTED OUT
        self.silu = nn.SiLU()  # REPLACEMENT

    def forward(self, x):
        # x: (1, 320)  -> for example

        x = self.linear_1(x)
        x = self.leaky(x)  # COMMENTED OUT
        x = self.silu(x)     # REPLACEMENT
        x = self.linear_2(x)
        x = self.leaky(x)  # If you had used self.leaky here, comment out similarly
        # x = self.silu(x)     # If you want a second activation11
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=640):
        super().__init__()
        # GN1
        self.groupnorm_feature = nn.GroupNorm(8, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        self.leaky = nn.LeakyReLU(0.3)  # COMMENTED OUT
        self.silu = nn.SiLU()            # REPLACEMENT

        #GN 2
        self.groupnorm_merged = nn.GroupNorm(8, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        residue = feature
        
        #GN 1
        # feature = self.groupnorm_feature(feature)
        feature = self.leaky(feature)  # COMMENTED OUT
        # feature = self.silu(feature)    # REPLACEMENT
        feature = self.conv_feature(feature)
        
        time = self.leaky(time)  # COMMENTED OUT
        # time = self.silu(time)    # REPLACEMENT
        time = self.linear_time(time)
        
        # Broadcast time over spatial dimensions
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # GN2
        merged = self.groupnorm_merged(merged)
        merged = self.leaky(merged)  # COMMENTED OUT
        # merged = self.silu(merged)      # REPLACEMENT
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):

    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        channels = n_head * n_embd
        
        # GN 3
        self.groupnorm = nn.GroupNorm(8, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        # Self-attention
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)

        # Feed-forward (GeGLU)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        # x: (Batch_Size, Features, Height, Width)
        residue_long = x

        # GN 3
        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # Flatten the spatial dimensions
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)  # (n, h*w, c)

        # ----- Self-Attention -----
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x = x + residue_short

        # ----- Feed-Forward (GeGLU) -----
        residue_short = x
        x = self.layernorm_2(x)

        # GeGLU
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x = x + residue_short

        # Reshape back to (Batch_Size, Features, Height, Width)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    """
    Allows passing the `time` embedding to ResidualBlocks
    while removing context usage entirely.
    """
    def forward(self, x, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (Batch_Size, 32, Height/8, Width/8) -> (Batch_Size, 320, Height/8, Width/8)
            SwitchSequential(nn.Conv2d(32, 320, kernel_size=3, padding=1)),
            
            # (Batch_Size, 320) -> ResBlock -> SelfAttention
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # Downsample
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # (320 -> 640) -> ResBlock -> SelfAttention
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # Downsample
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # (640 -> 1280) -> ResBlock -> SelfAttention
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # Downsample
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # Deeper ResBlock chain
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )
        
        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, time):
        # x: (Batch_Size, 32, Height/8, Width/8)
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, time)
            skip_connections.append(x)

        x = self.bottleneck(x, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, time)
        
        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # GN 4
        self.groupnorm = nn.GroupNorm(8, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.leaky = nn.LeakyReLU(0.3)  # COMMENTED OUT
        # self.silu = nn.SiLU()            # REPLACEMENT

    def forward(self, x):
        # GN 4
        x = self.groupnorm(x)
        x = self.leaky(x)  # COMMENTED OUT
        # x = self.silu(x)     # REPLACEMENT
        x = self.conv(x)
        return x


class Diffusion(nn.Module):
    """
    Diffusion model without cross-attention/context.
    """
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 32)
    
    def forward(self, latent, time):
        """
        :param latent: (Batch_Size, 4=32, Height/8, Width/8)
        :param time:   (1, 320)
        """
        time = self.time_embedding(time)
        output = self.unet(latent, time)
        output = self.final(output)
        return output
