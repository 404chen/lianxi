import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DeconvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.block(x)


class AnomalyCAE_Final(nn.Module):
    """
    输入:  [B, 1, 1024, 32]
    输出:  [B, 1, 1024, 32]

    面向重建式异常检测:
    - 无 skip connection
    - 有 bottleneck 压缩
    - 有 dropout
    - 解码器容量受限
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=16,
        latent_dim=128,
        dropout=0.1
    ):
        super(AnomalyCAE_Final, self).__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, stride=2, dropout=0.0)              # 1024x32 -> 512x16
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2, dropout=0.0)        # 512x16  -> 256x8
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2, dropout=dropout) # 256x8   -> 128x4
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 4, stride=2, dropout=dropout) # 128x4   -> 64x2

        # 压缩到向量 bottleneck
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(base_channels * 4 * 64 * 2, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, base_channels * 4 * 64 * 2)

        # latent 正则
        self.latent_dropout = nn.Dropout(dropout)

        # Decoder
        self.dec1 = DeconvBlock(base_channels * 4, base_channels * 4, dropout=dropout)   # 64x2   -> 128x4
        self.dec2 = DeconvBlock(base_channels * 4, base_channels * 2, dropout=dropout)   # 128x4  -> 256x8
        self.dec3 = DeconvBlock(base_channels * 2, base_channels, dropout=0.0)            # 256x8  -> 512x16
        self.dec4 = DeconvBlock(base_channels, base_channels // 2, dropout=0.0)           # 512x16 -> 1024x32

        # 输出头尽量简单，避免解码器过强
        self.out_conv = nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1)

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.flatten(x)
        z = self.fc_enc(x)
        z = self.latent_dropout(z)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(z.size(0), -1, 64, 2)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        recon = self.out_conv(x)
        return recon

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon