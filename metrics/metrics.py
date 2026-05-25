
import math
from typing import Union, Tuple, List

import torch

from metrics import flops


@torch.no_grad()
def get_flops(model, x_input):
    return flops.flops(model, x_input)


@torch.no_grad()
def get_theoretical_speedup(n_flops: int, n_nonzero_flops: int) -> dict:
    if n_nonzero_flops == 0:
        # Would yield infinite speedup
        return {}
    return float(n_flops) / n_nonzero_flops


@torch.no_grad()
def get_parameter_count(model: torch.nn.Module) -> Tuple[int, int]:
    n_total = 0
    n_nonzero = 0
    param_list = ["weight", "bias"]
    for name, module in model.named_modules():
        for param_type in param_list:
            if hasattr(module, param_type) and not isinstance(
                getattr(module, param_type), type(None)
            ):
                p = getattr(module, param_type)
                n_total += int(p.numel())
                n_nonzero += int(torch.sum(p != 0))
    return n_total, n_nonzero


@torch.no_grad()
def get_distance_to_origin(model: torch.nn.Module) -> float:
    prune_vector = torch.cat(
        [
            module.weight.flatten()
            for name, module in model.named_modules()
            if hasattr(module, "weight")
            and not isinstance(module.weight, type(None))
            and not isinstance(module, torch.nn.BatchNorm2d)
        ]
    )
    return float(torch.norm(prune_vector, p=2))
