from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

class ConvAP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        s1: int = 2,
        s2: int = 2,
    ) -> None:
        """
        :param in_channels: Number of channels in the input of ConvAP.
        :param out_channels: Number of channels that ConvAP outputs.
        :param s1: Spatial height of the adaptive average pooling.
        :param s2: Spatial width of the adaptive average pooling.
        """
        super().__init__()
        self.channel_pool = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.AAP = nn.AdaptiveAvgPool2d((s1, s2))

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_pool(x)
        x = self.AAP(x)
        x = x.flatten(1)
        x = F.normalize(x, p=2, dim=1)
        return x

