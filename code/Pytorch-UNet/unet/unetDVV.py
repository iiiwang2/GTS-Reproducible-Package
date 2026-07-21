import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

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
    

class VerticalAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.v_filter = nn.Conv2d(in_ch, 1, kernel_size=(7, 1), padding=(3, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v_map = self.sigmoid(self.v_filter(x)) 
        return x * v_map

# Up-conv
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # self.conv0 = ConvBlock(in_ch, in_ch)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
        self.vert_attn = VerticalAttention(out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2,
                        diffY//2, diffY-diffY//2])
        out = self.conv(torch.cat([x2, x1], dim=1))
        return self.vert_attn(out)
    
class DVVBlock(nn.Module):
    def __init__(self, in_ch, reduction=1, dilations=(2, 4, 8)):
        super().__init__()
        mid_ch = in_ch // reduction

        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )

        self.v_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, kernel_size=(k, 1),
                          dilation=(d, 1), padding=(d, 0)),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True)
            )
            for k, d in zip([3, 3, 3], dilations)
        ])
        
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * len(dilations), in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x_reduced = self.reduce(x)
        outs = [conv(x_reduced) for conv in self.v_convs]
        out = torch.cat(outs, dim=1)
        return self.fuse(out)


# HF Layer
class HF_Layer(nn.Module):
    def __init__(self, in_ch_list, out_ch):
        super().__init__()
        total_in = sum(in_ch_list)
        self.conv = nn.Conv2d(total_in, out_ch, kernel_size=1)

    def forward(self, feat_list):
        target_size = feat_list[-1].size()[2:]  
        up_feats = [F.interpolate(f, target_size, mode='bilinear', align_corners=True)
                    for f in feat_list]
        fused = torch.cat(up_feats, dim=1)
        return self.conv(fused)

    

# UnetDVH-Linear 主体
class UnetDVVLinear(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, use_dvh_in_encoder=False, use_dvh_after_encoder=True):
        super().__init__()
        self.use_dvh_in_encoder = use_dvh_in_encoder
        self.use_dvh_after_encoder = use_dvh_after_encoder

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = ConvBlock(n_channels, 64)
        self.down1c = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 64))
        self.down1d = nn.Sequential(nn.MaxPool2d(2), DVVBlock(64))

        self.down2c = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 128))
        self.down2d = nn.Sequential(nn.MaxPool2d(2), DVVBlock(128))

        self.down3c = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 256))
        self.down3d = nn.Sequential(nn.MaxPool2d(2), DVVBlock(256))

        self.down4c = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, 512))
        self.down4d = nn.Sequential(nn.MaxPool2d(2), DVVBlock(512))

        self.dvh_after = DVVBlock(1024)
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
        self.inc = CheckpointModule(self.inc)
        self.down1c = CheckpointModule(self.down1c)
        self.down1d = CheckpointModule(self.down1d)
        self.down2c = CheckpointModule(self.down2c)
        self.down2d = CheckpointModule(self.down2d)
        self.down3c = CheckpointModule(self.down3c)
        self.down3d = CheckpointModule(self.down3d)
        self.down4c = CheckpointModule(self.down4c)
        self.down4d = CheckpointModule(self.down4d)
        self.dvh_after = CheckpointModule(self.dvh_after)
        self.conv_after = CheckpointModule(self.conv_after)
        self.up1 = CheckpointModule(self.up1)
        self.up2 = CheckpointModule(self.up2)
        self.up3 = CheckpointModule(self.up3)
        self.up4 = CheckpointModule(self.up4)
        self.hf = CheckpointModule(self.hf)
        self.outc = CheckpointModule(self.outc)


class CheckpointModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *inputs):
        return cp.checkpoint(self.module, *inputs)