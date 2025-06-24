from pathlib import Path
import torch
from .qlora import QLoRABigNet  

def load(path: Path | None):
    # Extra credit: A BigNet variant that uses <4 bits/parameter and retains decent accuracy
    # A <9MB compressed model using 4-bit quantization + LoRA
    model = QLoRABigNet(
        lora_dim=2,        # reduce to smallest possible
        group_size=128     # increase group size
    )

    if path is not None:
        model.load_state_dict(torch.load(path, weights_only=True), strict=False)

    return model


