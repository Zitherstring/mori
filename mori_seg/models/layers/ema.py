# Copyright (c) OpenMMLab. All rights reserved.
"""ExpMomentumEMA variant used by MORI-seg.

Stock mmdet updates the average with ``mul_(1 - m).add_(src, alpha=m)``;
this variant uses ``lerp_(src, m)``. The two are mathematically equivalent
but not bit-identical in floating point, and the EMA weights are what gets
validated and saved. Registered as ``MORIExpMomentumEMA``.
"""
import math

from torch import Tensor

from mmdet.models.layers import ExpMomentumEMA
from mmdet.registry import MODELS


@MODELS.register_module()
class MORIExpMomentumEMA(ExpMomentumEMA):
    """ExpMomentumEMA with a ``lerp_`` update."""

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        momentum = (1 - self.momentum) * math.exp(
            -float(1 + steps) / self.gamma) + self.momentum
        averaged_param.lerp_(source_param, momentum)
