from dataclasses import dataclass
from pathlib import Path
from config import DataConfig

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    """Sliding windows over the traffic history. Samples are [T, N, C]."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        df = pd.read_hdf(config.data_path)
        self.data: Tensor = torch.from_numpy(df.to_numpy(dtype="float32", copy=True)).unsqueeze(-1)

    def __len__(self) -> int:
        window = self.config.seq_len + self.config.pred_len
        return (self.data.shape[0] - window) // self.config.stride + 1

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        start = idx * self.config.stride
        split = start + self.config.seq_len
        end = split + self.config.pred_len
        return {"x": self.data[start:split], "y": self.data[split:end]}

if __name__ == "__main__":
    config = DataConfig(data_path = "data/sd_his_2019.h5", seq_len = 12, pred_len = 1)
    ds = TrafficDataset(config)
    print(ds[1]["x"])
    print(ds[1]["x"].shape)
