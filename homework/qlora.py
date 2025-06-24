from pathlib import Path

import torch
import torch.nn as nn
from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit, block_dequantize_4bit
import math

class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)

        # TODO: Implement LoRA, initialize the layers, and make sure they are trainable
        # Keep the LoRA layers in float32
        # LoRA adapters (float32 and trainable)
        self.lora_a = nn.Linear(in_features, lora_dim, bias=False).float()
        self.lora_b = nn.Linear(lora_dim, out_features, bias=False).float()

        # Weight initialization
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        # Enable gradients for LoRA
        for p in self.lora_a.parameters():
            p.requires_grad = True
        for p in self.lora_b.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Forward. Make sure to cast inputs to self.linear_dtype and the output back to x.dtype
        input_dtype = x.dtype

        # Dequantize frozen base weights
        weight = block_dequantize_4bit(self.weight_q4, self.weight_norm)
        weight = weight.view(self._shape)

        # Linear using dequantized weights
        base_out = torch.nn.functional.linear(x, weight, self.bias)

        # LoRA residual (in float32)
        lora_out = self.lora_b(self.lora_a(x.float()))

        return (base_out + lora_out).to(input_dtype)


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
