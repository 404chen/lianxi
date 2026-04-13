import torch
import torch.nn as nn

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(ConvBlock1D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.block(x)

class DeconvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DeconvBlock1D, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.block(x)

class AnomalyCAE_Final_1D(nn.Module):
    """
    输入:  [B, 1, 1024]
    输出:  [B, 1, 1024]
    面向单根 A扫数据的 1D 异常检测重建模型
    """
    def __init__(self, in_channels=1, base_channels=16, latent_dim=64, dropout=0.1):
        super(AnomalyCAE_Final_1D, self).__init__()

        # Encoder (长度变化：1024 -> 512 -> 256 -> 128 -> 64)
        self.enc1 = ConvBlock1D(in_channels, base_channels, stride=2, dropout=0.0)
        self.enc2 = ConvBlock1D(base_channels, base_channels * 2, stride=2, dropout=0.0)
        self.enc3 = ConvBlock1D(base_channels * 2, base_channels * 4, stride=2, dropout=dropout)
        self.enc4 = ConvBlock1D(base_channels * 4, base_channels * 4, stride=2, dropout=dropout)

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(base_channels * 4 * 64, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, base_channels * 4 * 64)
        self.latent_dropout = nn.Dropout(dropout)

        # Decoder (长度变化：64 -> 128 -> 256 -> 512 -> 1024)
        self.dec1 = DeconvBlock1D(base_channels * 4, base_channels * 4, dropout=dropout)
        self.dec2 = DeconvBlock1D(base_channels * 4, base_channels * 2, dropout=dropout)
        self.dec3 = DeconvBlock1D(base_channels * 2, base_channels, dropout=0.0)
        self.dec4 = DeconvBlock1D(base_channels, base_channels // 2, dropout=0.0)

        self.out_conv = nn.Conv1d(base_channels // 2, 1, kernel_size=3, padding=1)

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
        # 恢复到序列形状 [B, Channels, Length]
        x = x.view(z.size(0), -1, 64)
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