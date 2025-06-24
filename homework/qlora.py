from pathlib import Path
import torch
import torch.nn as nn
import math

from .bignet import BIGNET_DIM, LayerNorm
from .low_precision import Linear4Bit, block_dequantize_4bit


class QLoRALinear(Linear4Bit):
    def __init__(self, in_features, out_features, lora_dim, group_size=4096, bias=False):
        super().__init__(in_features, out_features, bias=bias, group_size=group_size)
        self.requires_grad_(False)

        self.lora_a = nn.Linear(in_features, lora_dim, bias=False).half()
        self.lora_b = nn.Linear(lora_dim, out_features, bias=False).half()

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward with dequantized weights
        weight = block_dequantize_4bit(self.weight_q4, self.weight_norm).view(self._shape)
        base_out = torch.nn.functional.linear(x, weight, self.bias)

        # Compute LoRA path in float16 but use float32 for gradients
        x_fp16 = x.to(torch.float16)
        lora_out = self.lora_b(self.lora_a(x_fp16)).to(x.dtype)

        return base_out + lora_out


class QLoRABigNet(nn.Module):
    class Block(nn.Module):
        def __init__(self, channels, lora_dim=None, group_size=4096, bias=False):
            super().__init__()
            if lora_dim is not None:
                self.model = nn.Sequential(
                    QLoRALinear(channels, channels, lora_dim, group_size, bias),
                    nn.ReLU(),
                    Linear4Bit(channels, channels, bias=bias, group_size=group_size),
                    nn.ReLU(),
                    Linear4Bit(channels, channels, bias=bias, group_size=group_size),
                )
            else:
                self.model = nn.Sequential(
                    Linear4Bit(channels, channels, bias=bias, group_size=group_size),
                    nn.ReLU(),
                    Linear4Bit(channels, channels, bias=bias, group_size=group_size),
                    nn.ReLU(),
                    Linear4Bit(channels, channels, bias=bias, group_size=group_size),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim=1, group_size=4096, bias=False):
        super().__init__()
        self.model = nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size, bias),   # Block 0 — LoRA enabled
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, None, group_size, bias),       # Block 1
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, None, group_size, bias),       # Block 2
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, None, group_size, bias),       # Block 3
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, None, group_size, bias),       # Block 4
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, None, group_size, bias),       # Block 5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    model = QLoRABigNet()
    if path is not None:
        model.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return model
