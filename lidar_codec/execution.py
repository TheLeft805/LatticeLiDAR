"""Finite execution policies for one frozen arena/GELU numerical model."""
import torch


# Output, transfer buffers, combined layer checks, borrowed coarse logits.
VARIANTS = {
    'baseline': (False, False, False, False),
    'output': (True, False, False, False),
    'transfer': (False, True, False, False),
    'checks': (False, False, True, False),
    'coarse': (False, False, False, True),
    'combined': (True, True, True, True),
}


class ExecutionPolicy:
    def __init__(self):
        self.select('baseline')
        self.payload = self.metadata = self.device_metadata = None

    def select(self, name):
        self.output, self.transfer, self.checks, self.coarse = VARIANTS[name]
        self.name = name

    def buffers(self, size, lanes):
        if self.payload is None or self.payload.numel() < size:
            self.payload = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        if self.metadata is None:
            self.metadata = torch.empty(1+2*lanes, dtype=torch.int32, pin_memory=True)
            self.device_metadata = torch.empty_like(self.metadata, device='cuda')
        if self.metadata.numel() != 1+2*lanes:
            raise ValueError('Entropy lane count changed')
        return self.payload[:size], self.metadata, self.device_metadata

    def upload(self, owner, parsed, extent):
        lanes = len(parsed.stacks)
        payload, metadata, device_metadata = self.buffers(lanes*extent, lanes)
        payload.zero_()
        host = payload.view(lanes, extent).numpy()
        import numpy as np
        for lane, stack in enumerate(parsed.stacks):
            host[lane, :len(stack.data)] = np.frombuffer(stack.data, np.uint8)
            metadata[1+lane] = stack.state
            metadata[1+lanes+lane] = len(stack.data)
        metadata[0] = 0
        owner.storage[:, :extent].copy_(payload.view(lanes, extent), non_blocking=True)
        device_metadata.copy_(metadata, non_blocking=True)
        owner.states.copy_(device_metadata[1:1+lanes])
        owner.positions.copy_(device_metadata[1+lanes:])

    def download(self, owner):
        lanes = len(owner.states)
        _, metadata, device_metadata = self.buffers(0, lanes)
        torch.cat((owner.error.reshape(1), owner.states, owner.positions), out=device_metadata)
        metadata.copy_(device_metadata, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        if int(metadata[0]):
            raise ValueError('Poisoned entropy frame')
        states = metadata[1:1+lanes].tolist()
        positions = metadata[1+lanes:].tolist()
        extent = max(positions, default=0)
        if extent < 0 or extent > owner.capacity or min(positions, default=0) < 0:
            raise ValueError('Invalid entropy output extent')
        payload, _, _ = self.buffers(lanes*extent, lanes)
        host = payload.view(lanes, extent)
        host.copy_(owner.storage[:, :extent], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return states, positions, host.numpy()
