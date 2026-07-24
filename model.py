"""DualUCNN network architecture."""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=kernel // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=True),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class FeatureUCNNBranch(nn.Module):
    """
    UCNN encoder-decoder that returns a feature map and an auxiliary image.

    The auxiliary image keeps the original residual-image supervision, while
    the feature map is what the fusion module actually consumes.
    """

    def __init__(self, in_ch=12, out_ch=12, base_ch=32):
        super().__init__()
        c = base_ch
        self.out_ch = out_ch

        self.enc0 = nn.Sequential(ConvRelu(in_ch, c), ResBlock(c))
        self.down1 = nn.Conv2d(c, c * 2, 2, stride=2, bias=True)
        self.enc1 = nn.Sequential(ConvRelu(c * 2, c * 2), ResBlock(c * 2))
        self.down2 = nn.Conv2d(c * 2, c * 4, 2, stride=2, bias=True)
        self.enc2 = nn.Sequential(ConvRelu(c * 4, c * 4), ResBlock(c * 4))
        self.down3 = nn.Conv2d(c * 4, c * 8, 2, stride=2, bias=True)

        self.mid = nn.Sequential(
            ConvRelu(c * 8, c * 8), ResBlock(c * 8),
            ResBlock(c * 8), ConvRelu(c * 8, c * 8),
        )

        self.up2 = nn.Conv2d(c * 8, c * 4, 1, bias=True)
        self.dec2 = nn.Sequential(ConvRelu(c * 8, c * 4), ResBlock(c * 4))
        self.up1 = nn.Conv2d(c * 4, c * 2, 1, bias=True)
        self.dec1 = nn.Sequential(ConvRelu(c * 4, c * 2), ResBlock(c * 2))
        self.up0 = nn.Conv2d(c * 2, c, 1, bias=True)
        self.dec0 = nn.Sequential(ConvRelu(c * 2, c), ResBlock(c))

        self.base_proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=True)
        self.aux_head = nn.Conv2d(c, out_ch, 1, bias=True)
        nn.init.zeros_(self.aux_head.weight)
        nn.init.zeros_(self.aux_head.bias)

    def interpolate_base(self, x):
        _, _, H, W = x.shape
        x_pool = F.max_pool2d(x, kernel_size=2, stride=1, padding=1)
        x_pool = x_pool[:, :, :H, :W]
        x_down = F.interpolate(x_pool, scale_factor=0.5,
                               mode='bilinear', align_corners=False)
        x_base = F.interpolate(x_down, size=(H, W),
                               mode='bilinear', align_corners=False)
        return self.base_proj(x_base)

    def forward(self, x):
        x_base = self.interpolate_base(x)

        f0 = self.enc0(x)
        f1 = self.enc1(self.down1(f0))
        f2 = self.enc2(self.down2(f1))
        m = self.mid(self.down3(f2))

        d = F.interpolate(m, f2.shape[2:], mode='bilinear', align_corners=False)
        d = self.dec2(torch.cat([self.up2(d), f2], 1))
        d = F.interpolate(d, f1.shape[2:], mode='bilinear', align_corners=False)
        d = self.dec1(torch.cat([self.up1(d), f1], 1))
        d = F.interpolate(d, f0.shape[2:], mode='bilinear', align_corners=False)
        feat = self.dec0(torch.cat([self.up0(d), f0], 1))

        aux_out = (x_base + self.aux_head(feat)).clamp(0., 1.)
        return feat, aux_out


class PolarFeatureBranch(nn.Module):
    
    def __init__(self, in_ch=12, out_ch=12, base_ch=32):
        super().__init__()
        if in_ch != 12 or out_ch != 12:
            raise ValueError("PolarFeatureBranch expects 12 input and output channels.")

        self.group_net = FeatureUCNNBranch(4, 4, base_ch=base_ch)
        self.merge_feat = nn.Sequential(
            ConvRelu(base_ch * 3, base_ch),
            ResBlock(base_ch),
        )
        self.aux_refine = nn.Sequential(
            ConvRelu(12, base_ch),
            ResBlock(base_ch),
            nn.Conv2d(base_ch, 12, 1, bias=True),
        )
        nn.init.zeros_(self.aux_refine[-1].weight)
        nn.init.zeros_(self.aux_refine[-1].bias)

    def forward(self, polar_input):
        feats = []
        aux = []
        for ci in range(3):
            group = polar_input[:, ci * 4:(ci + 1) * 4]
            feat_i, aux_i = self.group_net(group)
            feats.append(feat_i)
            aux.append(aux_i)

        polar_feat = self.merge_feat(torch.cat(feats, dim=1))
        polar_aux = torch.cat(aux, dim=1)
        polar_aux = (polar_aux + self.aux_refine(polar_aux)).clamp(0., 1.)
        return polar_feat, polar_aux


class FeatureFusionBlock(nn.Module):
    """
    Feature-level fusion.

    The first fusion operation is concat(rgb_feat, polar_feat). The RGB
    auxiliary image is only used as a stable residual base for the final image.
    """

    def __init__(self, feat_ch=32, out_ch=12, mid_ch=32):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvRelu(feat_ch * 2, mid_ch),
            ResBlock(mid_ch),
            ResBlock(mid_ch),
        )
        self.gate = nn.Conv2d(mid_ch, out_ch, 1, bias=True)
        self.head = nn.Conv2d(mid_ch, out_ch, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, rgb_feat, polar_feat, rgb_base):
        x = self.fuse(torch.cat([rgb_feat, polar_feat], dim=1))
        gate = torch.sigmoid(self.gate(x))
        residual = self.head(x) * gate
        return (rgb_base + residual).clamp(0., 1.)


class DualUCNN(nn.Module):
    def __init__(self, in_ch=12, out_ch=12,
                 rgb_base_ch=32, polar_base_ch=32, fusion_mid_ch=32):
        super().__init__()
        if rgb_base_ch != polar_base_ch:
            raise ValueError("v2 feature fusion expects rgb_base_ch == polar_base_ch.")

        self.rgb_net = FeatureUCNNBranch(in_ch, out_ch, base_ch=rgb_base_ch)
        self.polar_net = PolarFeatureBranch(in_ch, out_ch, base_ch=polar_base_ch)
        self.fusion = FeatureFusionBlock(feat_ch=rgb_base_ch, out_ch=out_ch,
                                         mid_ch=fusion_mid_ch)

    def forward(self, rgb_input: torch.Tensor, polar_input: torch.Tensor):
        rgb_feat, rgb_out = self.rgb_net(rgb_input)
        polar_feat, polar_out = self.polar_net(polar_input)
        fused_out = self.fusion(rgb_feat, polar_feat, rgb_out)
        return rgb_out, polar_out, fused_out

    def rgb_params(self):
        return list(self.rgb_net.parameters())

    def polar_params(self):
        return list(self.polar_net.parameters())

    def fusion_params(self):
        return list(self.fusion.parameters())


