"""Owned output views for mutually exclusive, sequential geometry graph replays."""
import math
import torch


class GeometryWorkspace:
    """Outputs survive this layer only; entropy descriptors must own their inputs."""

    def __init__(self, maximum):
        self.maximum = maximum
        self.storage = {}

    def store(self, value, capacity, path=()):
        if value is None:
            return None
        if isinstance(value, dict):
            return {key:self.store(item, capacity, path+(key,)) for key,item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self.store(item, capacity, path+(index,)) for index,item in enumerate(value))
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'Unexpected geometry output at {path}')
        target = self.view(value.shape, value.dtype, value.device, capacity, path)
        if target.data_ptr() != value.data_ptr() or target.stride() != value.stride():
            target.copy_(value)
        return target

    def view(self, shape, dtype, device, capacity, path):
        """Owned sequential-replay view; the caller must overwrite every element."""
        if path not in self.storage:
            allocated = tuple(self.maximum if size == capacity else size for size in shape)
            self.storage[path] = torch.empty(math.prod(allocated), dtype=dtype, device=device)
        flat = self.storage[path]
        count = math.prod(shape)
        if flat.dtype != dtype or flat.numel() < count:
            raise ValueError(f'Inconsistent geometry workspace at {path}')
        return flat[:count].view(shape)



class NeuralWorkspace:
    """One slot per output role; callers clone delayed checks/CDFs before replay."""

    def __init__(self, maximum, device):
        self.work = GeometryWorkspace(maximum)
        self.prior = torch.empty(((maximum+31)//32, 4), dtype=torch.int32, device=device)

    def mixture(self, delta, prior, capacity):
        copied_delta = self.work.store(delta, capacity, ('delta',))
        copied_prior = self.prior[:len(prior)]
        copied_prior.copy_(prior)
        return copied_delta, copied_prior

    def result(self, k, capacity, output):
        return self.work.store(output, capacity, ('terminal' if k == 6 else 'prediction',))
