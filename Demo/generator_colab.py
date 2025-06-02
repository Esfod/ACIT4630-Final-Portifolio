"""
Generator Architecture
"""



import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_c, out_c)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # upsample
        if x1.size() != x2.size():
            x1 = torch.nn.functional.interpolate(x1,
                                                 size=(x2.size(2),
                                                       x2.size(3)),
                                                 mode="bilinear",
                                                 align_corners=False
                                                 )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self, in_channels=30, out_channels=3):
        super().__init__()
        self.input_block = ConvBlock(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.final_up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.input_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_bottleneck = self.bottleneck(x4)
        x = self.up1(x_bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.final_up(x)
        return torch.tanh(self.output_conv(x))


