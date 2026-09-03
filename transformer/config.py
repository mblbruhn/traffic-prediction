from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, kw_only=True)
class DataConfig:
    data_path: Path
    seq_len: int
    pred_len: int
    batchsize: int = 256
    stride: int = 1