import torch
from torch import Tensor
import torch.nn as nn

DINOv2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
} # 对应的特征维度

class DINOv2(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        num_trainable_blocks: int = 2,
        norm_layer: bool = False,
        return_token: bool = False,
        vggt_long_config: dict = None,
    ):
        """
        DINOv2 model
        :param model_name: ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        :param num_trainable_blocks: The number of last blocks in the model that are trainable.
        :param norm_layer: If True, a normalization layer is applied in the forward pass.
        :param return_token: If True, the forward pass returns both the feature map and the token.
        """
        super().__init__()
        assert model_name in DINOv2_ARCHS.keys(), f"Unknown model name: {model_name}"
        self.model = torch.hub.load("./LoopModels/dinov2", model_name, source='local', pretrained=False)
        self.model.load_state_dict(torch.load(vggt_long_config['Weights']['DINO']))
        self.num_channels = DINOv2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

        print('DINOv2 model loaded.')

    def forward(self, x: Tensor):
        """
        The forward method for the DINOv2 class.
        :param x: The input tensor [B, 3, H, W]. H and W should be divisible by 14.
        :return:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """
        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)

        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trainable
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        t = x[:, 0]
        f = x[:, 1:]
        f = f.reshape(B, H // 14, W // 14, self.num_channels).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        else:
            return f
