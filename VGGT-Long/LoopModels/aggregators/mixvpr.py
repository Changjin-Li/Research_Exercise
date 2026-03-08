from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio = 1) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mix(x)

class MixVPR(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 512,
        in_h: int = 20,
        in_w: int = 20,
        mix_depth: int = 1,
        mlp_ratio: int = 1,
        out_rows: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_h = in_h
        self.in_w = in_w
        self.mix_depth = mix_depth
        self.mlp_ratio = mlp_ratio
        self.out_rows = out_rows

        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=self.in_h * self.in_w, mlp_ratio=self.mlp_ratio) for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(self.in_channels, self.out_channels)
        self.row_proj = nn.Linear(self.in_h * self.in_w, self.out_rows)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = x.flatten(1)
        x = F.normalize(x, p=2, dim=-1)
        return x