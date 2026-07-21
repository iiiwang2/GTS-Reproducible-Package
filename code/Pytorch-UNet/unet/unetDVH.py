import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from .posemb import SinusoidalPositionalConv, RoPE

# Conv 3*3 Relu
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Up-conv
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # self.conv0 = ConvBlock(in_ch, in_ch)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2,
                        diffY//2, diffY-diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))
    
# DVH Block
class DVHBlock(nn.Module):
    def __init__(self, in_ch, reduction=1, dilation=2):
        super().__init__()
        mid_ch = in_ch // reduction

        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
       
        self.v_conv = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(3,1),
                      dilation=(dilation,1), padding=(dilation,0)),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        self.h_conv = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(1,3),
                      dilation=(1,dilation), padding=(0,dilation)),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )

        self.bn = nn.BatchNorm2d(mid_ch*2)
        self.relu = nn.ReLU(inplace=True)
        self.fuse = nn.Conv2d(mid_ch*2, in_ch, kernel_size=1)

    def forward(self, x):
        x_reduced = self.reduce(x)
        v = self.v_conv(x_reduced)
        h = self.h_conv(x_reduced)
        out = torch.cat([v, h], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return self.fuse(out)

# HF Layer
class HF_Layer(nn.Module):
    def __init__(self, in_ch_list, out_ch):
        super().__init__()
        total_in = sum(in_ch_list)
        self.conv = nn.Conv2d(total_in, out_ch, kernel_size=1)

    def forward(self, feat_list):
        target_size = feat_list[-1].size()[2:]  
        # Use NCHW for interpolate to avoid NHWC INT_MAX limitation
        up_feats = [
            F.interpolate(
                f.contiguous(memory_format=torch.contiguous_format),
                target_size,
                mode='bilinear',
                align_corners=True
            )
                    for f in feat_list]
        fused = torch.cat(up_feats, dim=1)
        return self.conv(fused)

    

# UnetDVH-Linear 主体
class UnetDVHLinear(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, use_dvh_in_encoder=False, use_dvh_after_encoder=True):
        super().__init__()
        self.use_dvh_in_encoder = use_dvh_in_encoder
        self.use_dvh_after_encoder = use_dvh_after_encoder

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = False

        self.inc = ConvBlock(n_channels, 64)
        self.down1c = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 64))
        self.down1d = nn.Sequential(nn.MaxPool2d(2), DVHBlock(64))

        self.down2c = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 128))
        self.down2d = nn.Sequential(nn.MaxPool2d(2), DVHBlock(128))

        self.down3c = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 256))
        self.down3d = nn.Sequential(nn.MaxPool2d(2), DVHBlock(256))

        self.down4c = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, 512))
        self.down4d = nn.Sequential(nn.MaxPool2d(2), DVHBlock(512))

        self.dvh_after = DVHBlock(1024)
        self.conv_after = ConvBlock(1024, 1024)

        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        self.hf = HF_Layer([1024, 512, 256, 128, 64], 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = torch.cat([self.down1c(x1), self.down1d(x1)], dim=1)
        x3 = torch.cat([self.down2c(x2), self.down2d(x2)], dim=1)
        x4 = torch.cat([self.down3c(x3), self.down3d(x3)], dim=1)
        x5 = torch.cat([self.down4c(x4), self.down4d(x4)], dim=1)
        
        x6 = self.dvh_after(x5)
        u0 = self.conv_after(x6)

        u1 = self.up1(u0, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)

        fused = self.hf([u0, u1, u2, u3, u4])
        return self.outc(fused)
    
    def use_checkpointing(self):
        # 用 checkpoint 包装模块以减少显存占用
        self.inc = cp.checkpoint_sequential(self.inc, segments=1)
        self.down1c = cp.checkpoint_sequential(self.down1c, segments=1)
        self.down1d = cp.checkpoint_sequential(self.down1d, segments=1)
        self.down2c = cp.checkpoint_sequential(self.down2c, segments=1)
        self.down2d = cp.checkpoint_sequential(self.down2d, segments=1)
        self.down3c = cp.checkpoint_sequential(self.down3c, segments=1)
        self.down3d = cp.checkpoint_sequential(self.down3d, segments=1)
        self.down4c = cp.checkpoint_sequential(self.down4c, segments=1)
        self.down4d = cp.checkpoint_sequential(self.down4d, segments=1)
        self.dvh_after = cp.checkpoint_sequential(self.dvh_after, segments=1)
        self.conv_after = cp.checkpoint_sequential(self.conv_after, segments=1)
        self.up1 = cp.checkpoint_sequential(self.up1, segments=1)
        self.up2 = cp.checkpoint_sequential(self.up2, segments=1)
        self.up3 = cp.checkpoint_sequential(self.up3, segments=1)
        self.up4 = cp.checkpoint_sequential(self.up4, segments=1)
        self.hf = cp.checkpoint_sequential(self.hf, segments=1)
        self.outc = cp.checkpoint_sequential(self.outc, segments=1)


class UnetDVHPosEmb(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, use_dvh_in_encoder=False, use_dvh_after_encoder=True):
        super().__init__()
        self.use_dvh_in_encoder = use_dvh_in_encoder
        self.use_dvh_after_encoder = use_dvh_after_encoder

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = False

        self.inc = ConvBlock(n_channels, 64)
        self.down1c = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 64))
        self.down1d = nn.Sequential(nn.MaxPool2d(2), DVHBlock(64))

        self.down2c = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 128))
        self.down2d = nn.Sequential(nn.MaxPool2d(2), DVHBlock(128))

        self.down3c = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 256))
        self.down3d = nn.Sequential(nn.MaxPool2d(2), DVHBlock(256))

        self.down4c = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, 512))
        self.down4d = nn.Sequential(nn.MaxPool2d(2), DVHBlock(512))

        self.dvh_after = DVHBlock(1024)
        self.pos_conv = SinusoidalPositionalConv(1024)
        self.rope = RoPE(1024)
        self.conv_after = ConvBlock(1024, 1024)

        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        self.hf = HF_Layer([1024, 512, 256, 128, 64], 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = torch.cat([self.down1c(x1), self.down1d(x1)], dim=1)
        x3 = torch.cat([self.down2c(x2), self.down2d(x2)], dim=1)
        x4 = torch.cat([self.down3c(x3), self.down3d(x3)], dim=1)
        x5 = torch.cat([self.down4c(x4), self.down4d(x4)], dim=1)

        x6 = self.dvh_after(x5)
        x6 = self.pos_conv(x6)
        x6 = self.rope(x6, "row")
        x6 = self.rope(x6, "col")
        u0 = self.conv_after(x6)

        u1 = self.up1(u0, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)

        fused = self.hf([u0, u1, u2, u3, u4])
        return self.outc(fused)
