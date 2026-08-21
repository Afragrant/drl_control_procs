"""History-window LSTM representation for LSTM-SAC."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from xuance.torch import Module
from xuance.torch.utils import ModuleType

LSTM_HIDDEN = 256


class HistoryLSTM(Module):
    """Encode a [batch, seq_len, obs_dim] observation window with a single-layer LSTM."""

    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_sizes: Sequence[int] | None = None,
        normalize: ModuleType | None = None,
        initialize: Callable[..., Tensor] | None = None,
        activation: ModuleType | None = None,
        device: str | int | torch.device | None = None,
        **kwargs,
    ):
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError(f'HistoryLSTM expects input_shape (seq_len, obs_dim), got {input_shape}')

        # seq_len：历史帧数, obs_dim: 单帧观测维度
        seq_len, input_size = int(input_shape[0]), int(input_shape[-1])
        if seq_len <= 0:
            raise ValueError(f'HistoryLSTM expects a positive sequence length, got {seq_len}')
        if input_size != 5:
            raise ValueError(f'HistoryLSTM expects obs_dim 5, got {input_size}')

        hidden_size = int(hidden_sizes[-1]) if hidden_sizes else LSTM_HIDDEN
        self.input_shape = (seq_len, input_size)
        self.device = device
        self.output_shapes = {'state': (hidden_size,)}
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        ).to(device)

    def forward(self, observations: Tensor) -> dict[str, Tensor]:
        seq_len, input_size = self.input_shape
        x = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(
                f'HistoryLSTM expects (batch, {seq_len}, {input_size}) or ({seq_len}, {input_size}), '
                f'got {tuple(x.shape)}'
            )
        if x.shape[-2:] != self.input_shape:
            raise ValueError(f'HistoryLSTM expects trailing shape {self.input_shape}, got {tuple(x.shape[-2:])}')
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        return {'state': output[:, -1, :]}
