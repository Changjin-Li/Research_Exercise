import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

class GeMPool(nn.Module):
    def __init__(self, p=3, eps=1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        x = x.flatten(1)
        x = F.normalize(x, p=2, dim=1)
        return x