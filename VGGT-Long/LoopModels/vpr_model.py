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

        self.backbone = utils.get_backbone(backbone_arch, backbone_config, vggt_long_config)
        self.aggregator = utils.get_aggregator(aggregator_arch, aggregator_config)
        self.val_outputs = []

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

    def configure_optimizers(self):
        if self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer} not implemented")

        if self.lr_scheduler.lower() == "linear":
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_scheduler_config['start_factor'],
                end_factor=self.lr_scheduler_config['end_factor'],
                total_iters=self.lr_scheduler_config['total_iters']
            )
        elif self.lr_scheduler.lower() == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_scheduler_config['T_max'])
        elif self.lr_scheduler.lower() == "multistep":
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_scheduler_config['milestones'], gamma=self.lr_scheduler_config['gamma'])
        else:
            raise NotImplementedError(f"Scheduler {self.lr_scheduler} not implemented")

        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()

    def loss_function(self, descriptors, labels):
        if self.miner is not None:
            miner_output = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_output)

            # Calculate the % of trivial pairs/triplets which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_output[0].detach().cpu().numpy()))
            batch_acc = 1.0 - nb_mined / nb_samples
        else:
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                loss, batch_acc = loss

        self.batch_acc.append(batch_acc)
        self.log('b_acc', sum(self.batch_acc) / len(self.batch_acc), prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        places, labels = batch
        BS, N, ch, H, W = places.shape
        images = places.view(BS * N, ch, H, W)
        labels = labels.view(-1)

        descriptors = self(images)
        if torch.isnan(descriptors).any():
            raise ValueError("NaN values in descriptors")
        loss = self.loss_function(descriptors, labels)
        self.log('loss', loss.item(), prog_bar=True, logger=True)
        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # We empty the batch_acc list for next epoch.
        self.batch_acc = []

    def validation_step(self, batch, batch_idx, dataloader_idx = None):
        places, _ = batch
        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()

    def on_validation_epoch_start(self) -> None:
        # reset the outputs list
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_dataloaders))]

    def on_validation_epoch_end(self) -> None:
        # [R1, R2, ..., Rn, Q1, Q2, ...]
        val_step_outputs = self.val_outputs
        dm = self.trainer.datamodule
        if len(dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)

            if "pitts" in val_set_name:
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                raise NotImplementedError(f'Please implement validation_epoch_end for {val_set_name}')

            r_list = feats[:num_references]
            q_list = feats[num_references:]
            pitts_dict = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 20, 30, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f"{val_set_name}/R1", pitts_dict[1], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R5", pitts_dict[5], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R10", pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')
        
        # reset the outputs list
        self.val_outputs = []

