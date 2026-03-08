import torch
from torch import Tensor
import torch.nn as nn
import torchvision

class ResNet(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        layers_to_freeze: int = 2,
        layers_to_crop: list = [],
    ):
        """
        We consider resnet network as a list of 5 blocks (from 0 to 4).
        Layer 0 is the first Conv+BN and the other layers (1 to 4) are the rest of the residual blocks.
        We don't take into account the Global-Pooling and the last FC.
        :param model_name: The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
        :param pretrained: Whether pretrained or not. Defaults to True.
        :param layers_to_freeze: The number of residual blocks to freeze (starting from 0) . Defaults to 2.
        :param layers_to_crop: Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].
        """
        super().__init__()
        self.model_name = model_name.lower() # 将大写字母改为小写字母
        self.layers_to_freeze = layers_to_freeze

        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None

        if 'swsl' in self.model_name or 'ssl' in self.model_name:
            self.model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', model_name)
        else:
            if 'resnext50' in model_name:
                self.model = torchvision.models.resnext50_32x4d(weights=weights)
            elif 'resnet50' in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif 'resnet101' in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif 'resnet152' in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif 'resnet34' in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif 'resnet18' in model_name:
                self.model = torchvision.models.resnet18(weights=weights)
            elif 'wide_resnet50_2' in model_name:
                self.model = torchvision.models.wide_resnet50_2(weights=weights)
            else:
                raise NotImplementedError('Backbone architecture not recognized!')

        # freeze only if the model is pretrained
        if pretrained:
            if self.layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if self.layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if self.layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if self.layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the AvgPool, FC layer and layers to crop
        self.model.avgpool = None
        self.model.fc = None
        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if 'resnet18' or 'resnet34' in self.model_name:
            out_channels = 512
        self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        self.out_channels = self.out_channels // 2 if self.model.layer3 is None else self.out_channels

    def forward(self, x: Tensor):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x