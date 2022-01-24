import os
import mat73
import numpy as np
from typing import Union
from wandb.sdk import wandb_run

import torch
from torch.utils import data
from torchvision import transforms as T

from .utils import plot_images


class NYUDataset(data.Dataset):
    def __init__(
        self,
        artifact_name: str,
        artifact_version: Union[str, int],
        run: Union[wandb_run.Run, None] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.run = run
        dataset_path = self._fetch_artifact(
            artifact_name=artifact_name, artifact_version=artifact_version
        )
        dataset_path = os.path.join(dataset_path, "nyu_depth_v2_labeled.mat")
        assert os.path.isfile(dataset_path), f"Unable to find Data file {dataset_path}"
        data_matrix = mat73.loadmat(dataset_path)
        self._titles = ["image", "depth_map", "segmentation_label"]
        self.images = data_matrix["images"]
        self.depths = data_matrix["depths"]
        self.labels = data_matrix["labels"]

    def _fetch_artifact(self, artifact_name, artifact_version):
        artifact = self.run.use_artifact(f"{artifact_name}:{str(artifact_version)}")
        return artifact.download()

    def __len__(self):
        return self.images.shape[-1]

    def __getitem__(self, idx):
        timg = T.ToTensor()(self.images[..., idx])
        tdepth = torch.as_tensor(self.depths[..., idx])
        tlabel = torch.as_tensor(self.labels[None, :, :, idx].astype(np.int64))
        return (timg, tdepth, tlabel)

    def show_one(self, idx=0):
        timg, tdepth, tlabel = self.__getitem__(idx)
        plot_images(
            [
                timg.permute(1, 2, 0).numpy(),
                tdepth.numpy(),
                tlabel.permute(1, 2, 0).numpy()[..., 0],
            ],
            titles=self._titles,
        )
