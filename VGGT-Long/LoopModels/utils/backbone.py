from LoopModels import backbones
import torch.nn as nn

def get_backbone(backbone_arch: str = 'resnet50', backbone_config = {}, vggt_long_config = None) -> nn.Module:
    if 'resnet' in backbone_arch.lower():
        return backbones.ResNet(backbone_arch, **backbone_config)
    if 'dinov2' in backbone_arch.lower():
        return backbones.DINOv2(model_name=backbone_arch, vggt_long_config=vggt_long_config, **backbone_config)
    raise NotImplementedError(f'Backbone {backbone_arch} not implemented')