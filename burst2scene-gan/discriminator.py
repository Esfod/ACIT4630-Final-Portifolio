"""
Discriminator Architecture
"""


import torch
import torch.nn as nn

class ConditionalDiscriminator(nn.Module):
    def __init__(self, in_channels=33):
        super().__init__()
        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)  # Output patch-level confidence map
        )

    def forward(self, burst, image):
        x = torch.cat([burst, image], dim=1)  # Shape: [B, 33, H, W]
        return self.model(x)  # Output: [B, 1, H', W']


