import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
import utils

class VPRModel(pl.LightningModule):
    """
    This is the main model for Visual Place Recognition.
    We use Pytorch Lightning for modularity purposes.
    Args:
        pl (_type_): _description_
    """
    def __init__(self,
        # Backbone
        backbone_arch: str = "resnet50",
        backbone_config = {},

        # Aggregator
        aggregator_arch: str = "ConvAP",
        aggregator_config = {},

        # Train hyperparameters
        lr: float = 0.03,
        optimizer: str = "sgd",
        weight_decay: float = 1e-3,
        momentum: float = 0.9,
        lr_scheduler: str = "linear",
        lr_scheduler_config = {
            "start_factor": 1,
            "end_factor": 0.2,
            "total_iters": 4000,
        },

        # Loss
        loss_name: str = "MultiSimilarityLoss",
        miner_name: str = 'MultiSimilarityMiner',
        miner_margin: float = 0.1,
        faiss_gpu: bool = False,
        vggt_long_config = None,
    ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config
        self.aggregator_arch = aggregator_arch
        self.aggregator_config = aggregator_config
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_config = lr_scheduler_config
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        self.save_hyperparameters() # write hyperparams into a file

        self.loss_fn = utils.get_loss(self.loss_name)
        self.miner = utils.get_miner(self.miner_name, self.miner_margin)
        self.batch_acc = [] # We will keep track of the % of trivial pairs/triplets at the loss level.
        self.faiss_gpu = faiss_gpu

