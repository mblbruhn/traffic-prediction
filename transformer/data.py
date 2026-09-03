from pathlib import Path

from config import DataConfig

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def _load_node_features(meta_path: Path) -> Tensor:
    """Static per-node features from sd_meta.csv, row order matches sd_his columns."""
    meta = pd.read_csv(meta_path)
    lat = ((meta["Lat"] - meta["Lat"].mean()) / meta["Lat"].std()).to_numpy(dtype="float32")
    lng = ((meta["Lng"] - meta["Lng"].mean()) / meta["Lng"].std()).to_numpy(dtype="float32")
    lanes = (meta["Lanes"] / meta["Lanes"].max()).to_numpy(dtype="float32")
    route = meta["Fwy"].str[:-2]  # "I5-N" -> "I5"; last 2 chars are the direction suffix
    route_onehot = pd.get_dummies(route).to_numpy(dtype="float32")
    direction_onehot = pd.get_dummies(meta["Direction"]).to_numpy(dtype="float32")
    features = np.concatenate(
        [lat[:, None], lng[:, None], lanes[:, None], route_onehot, direction_onehot], axis=1
    )
    return torch.from_numpy(features)


class TrafficDataset(Dataset):
    """Sliding windows over the traffic history. Samples are [T, N, C]."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        df = pd.read_hdf(config.data_path)
        self.data: Tensor = torch.from_numpy(df.to_numpy(dtype="float32", copy=True)).unsqueeze(-1)
        self.node_features: Tensor = _load_node_features(config.meta_path)

    def __len__(self) -> int:
        window = self.config.seq_len + self.config.pred_len
        return (self.data.shape[0] - window) // self.config.stride + 1

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        start = idx * self.config.stride
        split = start + self.config.seq_len
        end = split + self.config.pred_len
        return {"x": self.data[start:split], "y": self.data[split:end], "node_features": self.node_features}


def make_dataloader(dataset: TrafficDataset, config: DataConfig, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batchsize,
        num_workers=config.num_workers,
        shuffle=shuffle,
    )


if __name__ == "__main__":
    config = DataConfig(data_path = "data/sd_his_2019.h5", meta_path = "data/sd_meta.csv", seq_len = 12, pred_len = 1)
    ds = TrafficDataset(config)
    print(ds[1]["x"])
    print("Shape of x: ", ds[1]["x"].shape)
    print("Shape of node features: ", ds[1]["node_features"].shape)
