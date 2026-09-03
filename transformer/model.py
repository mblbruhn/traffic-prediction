import lightning as L
import torch
from torch import Tensor, nn


class TrafficTransformer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(num_nodes, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, num_nodes)
        self.horizon_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: Tensor) -> Tensor:
        x = x.squeeze(-1)  # [B, T, N]
        h = self.encoder(self.input_proj(x))  # [B, T, d_model]
        out = self.output_proj(h)  # [B, T, N]
        out = self.horizon_proj(out.transpose(1, 2))  # [B, N, pred_len]
        return out.transpose(1, 2).unsqueeze(-1)  # [B, pred_len, N, 1]


class TrafficForecaster(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3) -> None:
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        y_hat = self(batch["x"])
        loss = nn.functional.mse_loss(y_hat, batch["y"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        y_hat = self(batch["x"])
        loss = nn.functional.mse_loss(y_hat, batch["y"])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
