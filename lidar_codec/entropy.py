"""Compact 32-lane ANS framing; state evolution lives in the CUDA backend."""

import struct
import zlib
from dataclasses import dataclass

LOWER = 1 << 23
LANES = 32
HEADER = struct.Struct("<4sBBBBII")


@dataclass
class LaneState:
    state: int
    data: bytes


def pack(stacks, *, active, carry, groups, symbols):
    body = bytearray(
        HEADER.pack(b"JSB1", 1, LANES, int(active), carry, groups, symbols)
    )
    for s in stacks:
        body += struct.pack("<II", s.state, len(s.data)) + s.data
    return bytes(body) + struct.pack("<I", zlib.crc32(body))


class EntropyFrame:
    def __init__(self, packet, carry):
        if len(packet) < HEADER.size + LANES * 8 + 4:
            raise ValueError("Truncated ANS frame")
        if (
            zlib.crc32(packet[:-4])
            != struct.unpack_from("<I", packet, len(packet) - 4)[0]
        ):
            raise ValueError("ANS checksum mismatch")
        magic, version, lanes, active, n, groups, symbols = HEADER.unpack_from(packet)
        if (magic, version, lanes, n) != (b"JSB1", 1, LANES, carry) or active not in (
            0,
            1,
        ):
            raise ValueError("ANS identity mismatch")
        if not 0 < groups <= 126 or not 0 < symbols <= 100_000_000:
            raise ValueError("ANS extent invalid")
        self.active, self.carry = active, carry
        self.expected = (groups, symbols)
        self.stacks = []
        at = HEADER.size
        for _ in range(LANES):
            if at + 8 > len(packet) - 4:
                raise ValueError("Truncated ANS lane header")
            state, size = struct.unpack_from("<II", packet, at)
            at += 8
            if not LOWER <= state < LOWER * 256 or at + size > len(packet) - 4:
                raise ValueError("ANS lane extent/state invalid")
            self.stacks.append(LaneState(state, packet[at : at + size]))
            at += size
        if at != len(packet) - 4:
            raise ValueError("ANS trailing bytes")
