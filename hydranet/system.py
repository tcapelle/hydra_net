import wandb
from tqdm import tqdm
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim

from .dataset import NYUDataset
from .model import HydraNet


class HydraNetSystem:
    def __init__(
        self, project_name: str = "hydranet", experiment_name: str = "experiment_1"
    ) -> None:
        wandb.login()
        self.run = wandb.init(project=project_name, experiment_name=experiment_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_loader(
        self,
        artifact_name: str,
        artifact_version: Union[str, int],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        self.train_dataset = NYUDataset(
            artifact_name=artifact_name, artifact_version=artifact_version, run=self.run
        )
        self.train_dataloader = self.train_dataset.build_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=pin_memory,
        )

    def compile(
        self,
        backbone: str = "mobilenetv2_100",
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        self.model = HydraNet(
            backbone=backbone,
            hidden_dim=hidden_dim,
            num_classes=self.train_dataset.num_classes,
        )
        self.optimizer = optim.Adam(
            params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        predictions = self.model(x)
        loss = self.criterion(predictions, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, epochs: int):
        for epoch in range(1, epochs + 1):
            print(f"Training Epoch ({epoch}/{epochs})...")
            progress_bar = tqdm(self.train_dataloader, leave=True)
            epoch_loss_history = []
            for x, y in progress_bar:
                epoch_loss = self.train_step(x, y)
                epoch_loss_history.append(epoch_loss)
                progress_bar.set_postfix(epoch_loss=epoch_loss)
            mean_epoch_loss = sum(epoch_loss_history) / len(epoch_loss_history)
            print(
                f"Completed training Epoch ({epoch}/{epochs}). Mean Loss={mean_epoch_loss}"
            )
            wandb.log({"mean_epoch_loss": mean_epoch_loss}, step=epoch)
    
    def infer(self, image):
        return self.model(image)
