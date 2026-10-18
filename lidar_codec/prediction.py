from typing import NamedTuple
import torch


class Prediction(NamedTuple):
    stage: int
    group: str
    rows: torch.Tensor
    logits: torch.Tensor
    cdf: object = None
    cdf_identity: object = None
    cdf_good: object = None
    delta: object = None
    prior: object = None
