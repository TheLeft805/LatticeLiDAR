"""Keep CUDA capture ordering, but collect cyclic garbage once per setup batch.

The default path still uses torch.cuda.graph. The optional path keeps its
device synchronization, allocator-cache release, stream context and capture
begin/end calls. Only explicit Python cycle collection is batched. Instances
are single-thread setup scopes; GC state is restored even on failure.
"""

import gc
import time
from contextlib import contextmanager
import torch


class CaptureSession:
    def __init__(self, policy="original"):
        if policy not in ("original", "once"):
            raise ValueError("Unknown capture GC policy")
        self.policy = policy
        self.stats = dict(policy=policy, graphs=0, collection_seconds=0.0)

    def collect(self):
        tick = time.perf_counter()
        gc.collect()
        self.stats["collection_seconds"] += time.perf_counter() - tick

    def __enter__(self):
        self.was_enabled = gc.isenabled()
        if self.policy == "once":
            torch.cuda.synchronize()
            gc.disable()
            try:
                self.collect()
            except BaseException:
                if self.was_enabled:
                    gc.enable()
                raise
        return self

    def __exit__(self, *exc):
        if self.policy == "once":
            try:
                torch.cuda.synchronize()
                self.collect()
            finally:
                if self.was_enabled:
                    gc.enable()

    @contextmanager
    def graph(self, graph, *, stream, pool=None):
        if self.policy == "original":
            with torch.cuda.graph(graph, stream=stream, pool=pool):
                yield
        else:
            # Match installed torch.cuda.graph.__enter__, excluding gc.collect.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            with torch.cuda.stream(stream):
                graph.capture_begin(pool=pool, capture_error_mode="global")
                try:
                    yield
                finally:
                    graph.capture_end()
        self.stats["graphs"] += 1
