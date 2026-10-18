"""Conservative per-lane bounds for the release's block-striped ANS encoder."""


def encoder_capacity(group_sizes, lanes=32):
    if lanes <= 0 or any(n <= 0 for n in group_sizes):
        raise ValueError('Expected positive lane and group sizes')
    capacity = 16
    for n in group_sizes:
        blocks = (n + 31)//32
        blocks_per_lane = (blocks + lanes - 1)//lanes
        # Each p16 symbol writes at most two bytes. The p7 prior writes at
        # most one, conservatively charged two. Posterior pops only shrink it.
        capacity += 2*(min(n, 32*blocks_per_lane) + blocks_per_lane)
    return capacity


def decoder_capacity(initial_extent, symbols, groups):
    if initial_extent < 0 or symbols < 0 or groups < 0:
        raise ValueError('Negative stream extent')
    # Decoding K4 writes up to two posterior bytes per block. This bound
    # deliberately gives every lane every block, so it needs no future CDFs
    # or unknown group lengths. Exact initial extent alone is NOT sufficient.
    return max(1, initial_extent + 2*(symbols//32 + groups))
