import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordDirectionalDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.coord_conv = CoordConv(in_channels, in_channels, kernel_size=1)
        self.directional_conv = DirectionalDoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.coord_conv(x)
        return self.directional_conv(x)

class DirectionalDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels

        # Horizontal Conv
        self.h_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )

        # Vertical Conv
        self.v_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )

        # Diagonal Conv
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )

        self.weight = nn.Parameter(torch.ones(3))  # [w_h, w_v, w_d]

        self.out_proj = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.h_conv(x)
        v = self.v_conv(x)
        d = self.d_conv(x)

        w = F.softmax(self.weight, dim=0)
        fused = w[0] * h + w[1] * v + w[2] * d

        out = self.out_proj(fused)
        out = out + self.residual_conv(x)
        return F.relu(out)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DirectionalDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DirectionalDoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DirectionalDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)

    def forward(self, x):
        batch_size, _, h, w = x.shape
        xx_channel = torch.linspace(-1, 1, w, device=x.device).repeat(h, 1).unsqueeze(0)
        yy_channel = torch.linspace(-1, 1, h, device=x.device).repeat(w, 1).t().unsqueeze(0)

        xx_channel = xx_channel.expand(batch_size, -1, -1, -1)
        yy_channel = yy_channel.expand(batch_size, -1, -1, -1)

        coord = torch.cat([x, xx_channel, yy_channel], dim=1)
        return self.conv(coord)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
