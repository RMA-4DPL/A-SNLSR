"""Credit to the original authors:

Qian Hu, Xinya Wang, Junjun Jiang, Xiao-Ping Zhang, and JiayiMa, “Exploring the spectral prior for hyperspectral image super-resolution,” IEEE Transactions on Image Processing, vol. 33, pp. 5260–5272, 2024.

Code from the original implementation: https://github.com/HuQ1an/SNLSR/
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def EzConv(in_channel, out_channel, kernel_size):
    return nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        bias=True,
    )


class SANL(nn.Module):

    def __init__(self, n_features: int):
        super(SANL, self).__init__()

        self.embedding = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1),
        )
        self.convq = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
        )
        self.convk = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
        )
        self.convv = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
        )
        self.to_out = nn.Linear(n_features, n_features)

    def forward(self, x):
        x = self.embedding(x)

        # input:[B,L,H,W], L:Num of Spectrals
        B, L, H, W = x.size()
        # x_re:[B,HW,l]
        x_re = x.view(B, L, H * W).permute(0, 2, 1)

        x_emb1 = self.convq(x_re)
        x_emb2 = self.convk(x_re)
        x_emb3 = self.convv(x_re)

        x_emb1 = F.normalize(x_emb1, dim=-1, p=2)
        x_emb2 = F.normalize(x_emb2, dim=-1, p=2)

        x_emb1 = torch.unsqueeze(x_emb1, dim=3)
        x_emb2 = torch.unsqueeze(x_emb2, dim=2)

        mat_product = torch.matmul(x_emb1, x_emb2)
        mat_product = F.softmax(mat_product, dim=3)

        x_emb3 = torch.unsqueeze(x_emb3, dim=3)

        attention = mat_product

        out = torch.matmul(attention, x_emb3)
        out = torch.squeeze(out, dim=3)
        out = self.to_out(out)

        out = out.permute(0, 2, 1).view(B, L, H, W)
        return out + x


class ESA(nn.Module):

    def __init__(self, n_features: int):
        super(ESA, self).__init__()
        f = n_features // 4
        self.conv1 = nn.Conv2d(n_features, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_features, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(
            c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class DDP(nn.Module):
    def __init__(self, n_features):
        super(DDP, self).__init__()
        self.pointconv = nn.Conv2d(n_features, n_features, 1)
        self.depthconv = nn.Conv2d(
            n_features, n_features, kernel_size=3, padding=1, groups=n_features
        )

    def forward(self, x, spex, spax):
        diffspex = self.pointconv(spex - x)
        diffspax = self.depthconv(spax - x)
        return x + diffspex + diffspax


class SSAM(nn.Module):
    def __init__(self, n_features, head, num):
        super(SSAM, self).__init__()
        self.ESA = ESA(n_features)
        self.NL = SANL(n_features)
        self.head = head
        if head == 0:
            self.botnek = nn.Conv2d(n_features * num, n_features, kernel_size=1)

        self.DDP = DDP(n_features)

    def forward(self, x):
        if self.head == 0:
            x = self.botnek(x)
        spex = self.NL(x)
        spax = self.ESA(x)
        out = self.DDP(x, spex, spax)
        return out


class Upsample(nn.Sequential):
    def __init__(self, scale, n_features, bn=False, act=False, bias=True, conv=EzConv):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_features, 4 * n_features, 3))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_features))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_features))

        elif scale == 3:
            m.append(conv(n_features, 9 * n_features, 3))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_features))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_features))
        else:
            raise NotImplementedError

        super(Upsample, self).__init__(*m)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) for channel and spatial attention.
    """

    def __init__(self, n_channels: int, reduction_ratio: int):
        super(CBAM, self).__init__()
        # Channel attention module
        self.mpl = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction_ratio, n_channels, bias=False),
        )

        # Spatial attention module
        self.conv = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False, padding_mode="reflect"
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_pool = F.avg_pool2d(X, X.size()[2:])
        max_pool = F.max_pool2d(X, X.size()[2:])

        # Forward through the shared MLP
        avg_out = self.mpl(avg_pool.view(X.size(0), -1)).view(
            X.size(0), X.size(1), 1, 1
        )
        max_out = self.mpl(max_pool.view(X.size(0), -1)).view(
            X.size(0), X.size(1), 1, 1
        )

        # Sum the outputs and apply channel attention
        X = X * torch.sigmoid(avg_out + max_out)

        # Spatial attention

        # Compute the average and max pooling along the channel dimension
        avg_out = torch.mean(X, dim=1, keepdim=True)
        max_out = torch.max(X, dim=1, keepdim=True)[0]

        # Concatenate the average and max pooling results and apply convolution
        out = torch.cat((avg_out, max_out), dim=1)
        out = self.conv(out)

        # Apply spatial attention
        return X * torch.sigmoid(out)


class SPDNEncoder(nn.Module):

    def __init__(
        self,
        input_channels: int,
        n_materials: int = 16,
    ):
        super(SPDNEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels, n_materials * 8, kernel_size=3, padding=1, bias=True
            ),
            CBAM(n_materials * 8, reduction_ratio=4),  # Channel and spatial attention
            nn.Tanh(),
            nn.Conv2d(
                n_materials * 8, n_materials * 4, kernel_size=3, padding=1, bias=True
            ),
            CBAM(n_materials * 4, reduction_ratio=4),  # Channel and spatial attention
            nn.Tanh(),
            nn.Conv2d(
                n_materials * 4, n_materials * 2, kernel_size=3, padding=1, bias=True
            ),
            CBAM(n_materials * 2, reduction_ratio=4),  # Channel and spatial attention
            nn.Tanh(),
            nn.Conv2d(n_materials * 2, n_materials, kernel_size=1, bias=True),
            nn.ReLU(),  # Avoind negative abundances
            nn.Softmax(dim=1),  # Ensure abundances sum to 1
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)
