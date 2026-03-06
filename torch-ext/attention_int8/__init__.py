from typing import Optional

import torch

from ._ops import ops


def attention_int8(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    ops.attention_int8(out, x)
    return out
