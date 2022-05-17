import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger


class LightningModel(LightningModule):

    def __init__(
        self,
        data_path: str="hand",
        arch: str = "resnet18",
        pretrained: bool = True,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 16,
        workers: int = 0,
    ):
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        self.train_acc1 = Accuracy(top_k=1)
        self.eval_acc1 = Accuracy(top_k=1)
        self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.model = models.__dict__[self.arch](pretrained=self.pretrained)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        loss_train = F.cross_entropy(output, target)
        self.log("train_loss", loss_train)
        # update metrics
        self.train_acc1(output, target)
        self.log("train_acc1", self.train_acc1, prog_bar=True)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self.model(images)
        loss_val = F.cross_entropy(output, target)
        self.log(f"{prefix}_loss", loss_val)
        # update metrics
        self.eval_acc1(output, target)
        self.log(f"{prefix}_acc1", self.eval_acc1, prog_bar=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 0.5 ** (epoch // 30))
        return [optimizer], [scheduler]



    def train_dataloader(self):
        train_dir = os.path.join(self.data_path, "train")
        self.train_dataset = datasets.ImageFolder(
                train_dir,
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        self.normalize,
                    ]
                ),
            )

        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_dir = os.path.join(self.data_path, "val")
        self.eval_dataset = datasets.ImageFolder(
            val_dir,
            transforms.Compose([transforms.Resize(224), transforms.CenterCrop(
                224), transforms.ToTensor(), self.normalize]),
        )

        return torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True
        )

    def test_dataloader(self):
        val_dir = os.path.join(self.data_path, "val")
        self.eval_dataset = datasets.ImageFolder(
            val_dir,
            transforms.Compose([transforms.Resize(224), transforms.CenterCrop(
                224), transforms.ToTensor(), self.normalize]),
        )

        return torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True
        )


model = LightningModel()

# GPU
# trainer = Trainer(
#     gpus=min(1, torch.cuda.device_count()),
#     max_epochs=90,
#     accelerator="gpu",
#     progress_bar_refresh_rate=10,
#     callbacks=[ModelCheckpoint(monitor='val_acc1',mode="max",save_top_k=1,save_weights_only=True)]
# )

#CPU
trainer = Trainer(
    max_epochs=90,
    accelerator="cpu",
    progress_bar_refresh_rate=5,
    callbacks=[ModelCheckpoint(monitor='val_acc1',mode="max",save_top_k=1,save_weights_only=True)]
)
trainer.fit(model)
trainer.test()
model.eval()
model.to_onnx("best.onnx", torch.randn((3, 224, 224)), export_params=True)