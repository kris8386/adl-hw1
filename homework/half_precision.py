from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm  # LayerNorm uses float32 internally


class HalfLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        # Convert weights and bias to half precision
        self.weight.data = self.weight.data.half()
        if self.bias is not None:
            self.bias.data = self.bias.data.half()
        # We don't need gradients
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_half = x.to(torch.float16)
        out_half = torch.nn.functional.linear(x_half, self.weight, self.bias)
        return out_half.to(input_dtype)


class HalfBigNet(torch.nn.Module):
    """
    A BigNet where all weights are in half precision.
    Normalization is kept in float32 for stability.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x  # Residual connection

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),   # Block 1
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),   # Block 2
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),   # Block 3
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),   # Block 4
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),   # Block 5
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),   # Block 6
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> HalfBigNet:
    # PyTorch can load float32 weights into float16 model layers if shapes match
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
