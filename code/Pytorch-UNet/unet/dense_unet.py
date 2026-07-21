import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SELU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.layer(x)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        self.out_channels = channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Dropout2d(dropout_rate),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.trans(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)


class DenseUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, growth_rate=16, num_layers=3):
        super().__init__()

        self.n_channels = in_channels
        self.n_classes = n_classes

        # 编码部分
        self.db1 = DenseBlock(in_channels, growth_rate, num_layers)
        self.trans1 = TransitionDown(self.db1.out_channels, self.db1.out_channels // 2)

        self.db2 = DenseBlock(self.db1.out_channels // 2, growth_rate, num_layers)
        self.trans2 = TransitionDown(self.db2.out_channels, self.db2.out_channels // 2)

        # self.db3 = DenseBlock(self.db2.out_channels // 2, growth_rate, num_layers)
        # self.trans3 = TransitionDown(self.db3.out_channels, self.db3.out_channels // 2)

        # self.db4 = DenseBlock(self.db3.out_channels // 2, growth_rate, num_layers)
        # self.trans4 = TransitionDown(self.db4.out_channels, self.db4.out_channels // 2)

        # 中间 bottleneck
        self.bottleneck = DenseBlock(self.db2.out_channels // 2, growth_rate, num_layers)

        # 解码部分
        # self.up4 = TransitionUp(self.bottleneck.out_channels, self.db4.out_channels // 2)
        # self.bottleneck4 = Bottleneck(self.db4.out_channels, self.db4.out_channels // 2)
        # self.db_up4 = DenseBlock(self.db4.out_channels, growth_rate, num_layers)

        # self.up3 = TransitionUp(self.db_up4.out_channels, self.db3.out_channels // 2)
        # self.bottleneck3 = Bottleneck(self.db3.out_channels, self.db3.out_channels // 2)
        # self.db_up3 = DenseBlock(self.db3.out_channels, growth_rate, num_layers)

        self.up2 = TransitionUp(self.bottleneck.out_channels, self.db2.out_channels // 2)
        self.bottleneck2 = Bottleneck(
            self.db2.out_channels // 2 + self.db2.out_channels, 
            self.db2.out_channels // 2
        )
        self.db_up2 = DenseBlock(self.db2.out_channels // 2, growth_rate, num_layers)

        self.up1 = TransitionUp(self.db_up2.out_channels, self.db1.out_channels // 2)
        self.bottleneck1 = Bottleneck(
            self.db1.out_channels // 2 + self.db1.out_channels, 
            self.db1.out_channels // 2
        )
        self.db_up1 = DenseBlock(self.db1.out_channels // 2, growth_rate, num_layers)

        # 最终输出
        self.final_conv = nn.Conv2d(self.db_up1.out_channels, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码路径
        x1 = self.db1(x)
        x1d = self.trans1(x1)

        x2 = self.db2(x1d)
        x2d = self.trans2(x2)

        # x3 = self.db3(x2d)
        # x3d = self.trans3(x3)

        # x4 = self.db4(x3d)
        # x4d = self.trans4(x4)

        # bottleneck
        xb = self.bottleneck(x2d)

        # 解码路径 + 跳跃连接
        # x = self.up4(xb)
        # x = self.bottleneck4(torch.cat([x, x4], dim=1))
        # x = self.db_up4(x)

        # x = self.up3(x)
        # x = self.bottleneck3(torch.cat([x, x3], dim=1))
        # x = self.db_up3(x)

        x = self.up2(xb)
        x = self.bottleneck2(torch.cat([x, x2], dim=1))
        x = self.db_up2(x)

        x = self.up1(x)
        x = self.bottleneck1(torch.cat([x, x1], dim=1))
        x = self.db_up1(x)

        return self.final_conv(x)
    
    def use_checkpointing(self):
        self.db1 = cp.checkpoint_sequential(self.db1.layers, len(self.db1.layers))
        self.trans1 = cp.checkpoint(self.trans1)

        self.db2 = cp.checkpoint_sequential(self.db2.layers, len(self.db2.layers))
        self.trans2 = cp.checkpoint(self.trans2)

        self.bottleneck = cp.checkpoint_sequential(self.bottleneck.layers, len(self.bottleneck.layers))

        self.up2 = cp.checkpoint(self.up2)
        self.bottleneck2 = cp.checkpoint(self.bottleneck2)
        self.db_up2 = cp.checkpoint_sequential(self.db_up2.layers, len(self.db_up2.layers))

        self.up1 = cp.checkpoint(self.up1)
        self.bottleneck1 = cp.checkpoint(self.bottleneck1)
        self.db_up1 = cp.checkpoint_sequential(self.db_up1.layers, len(self.db_up1.layers))

        self.final_conv = cp.checkpoint(self.final_conv)






