from dataclasses import dataclass
import struct

TR_MAGIC = b"ETR1"


@dataclass(frozen=True)
class TemporalHeader:
    previous_digest: bytes = b""
    pose: tuple = ()

    @property
    def reset(self):
        return not self.previous_digest

    def pack(self):
        if self.reset:
            if self.pose:
                raise ValueError("A reset cannot carry a reference pose")
            return TR_MAGIC + b"\x00"
        if len(self.previous_digest) != 16 or len(self.pose) != 6:
            raise ValueError(
                "TR requires a 16-byte reference digest and six quantized pose values"
            )
        return (
            TR_MAGIC + b"\x01" + self.previous_digest + struct.pack("<6i", *self.pose)
        )


@dataclass(frozen=True)
class FramePacket:
    posQ: int
    base_coords: tuple
    indices: tuple
    arithmetic: bytes
    temporal: object = None

    def _validate(self):
        if not self.arithmetic:
            raise ValueError("Arithmetic payload must not be empty")
        if not 1 <= self.posQ <= 32767 or not 1 <= len(self.base_coords) <= 64:
            raise ValueError("Invalid geometry header")
        if not 1 <= len(self.indices) <= 21 or any(
            not 0 <= index <= 255 for index in self.indices
        ):
            raise ValueError("Invalid expert index list")
        if any(
            len(row) != 3 or any(not 0 <= value <= 3 for value in row)
            for row in self.base_coords
        ):
            raise ValueError("Base support must lie in the original two-bit grid")

    def pack(self):
        self._validate()
        prefix = b"" if self.temporal is None else self.temporal.pack()
        coords = bytes(value for row in self.base_coords for value in row)
        return (
            prefix
            + struct.pack("<hB", self.posQ, len(self.base_coords))
            + coords
            + struct.pack("<I", len(self.arithmetic))
            + self.arithmetic
            + bytes(self.indices)
        )

    @property
    def bit_ledger(self):
        ledger = {
            "geometry_header_bits": (3 + 3 * len(self.base_coords) + 4) * 8,
            "expert_index_bits": len(self.indices) * 8,
            "arithmetic_bits": len(self.arithmetic) * 8,
            "temporal_bits": (
                0 if self.temporal is None else len(self.temporal.pack()) * 8
            ),
        }
        ledger["total_bits"] = sum(ledger.values())
        return ledger


def unpack_frame(data, *, temporal=False):
    """Parse before decoding; frame configuration is supplied by the matched checkpoint."""
    offset = 0
    condition = None
    if temporal:
        if len(data) < 5 or data[:4] != TR_MAGIC or data[4] not in (0, 1):
            raise ValueError("Missing or invalid TR frame prefix")
        offset = 5
        if data[4]:
            if len(data) < 45:
                raise ValueError("Truncated TR pose")
            condition = TemporalHeader(data[5:21], struct.unpack_from("<6i", data, 21))
            offset = 45
        else:
            condition = TemporalHeader()
    if len(data) < offset + 7:
        raise ValueError("Truncated ELiC frame")
    posQ, count = struct.unpack_from("<hB", data, offset)
    begin = offset + 3
    end = begin + count * 3
    if not 1 <= count <= 64 or len(data) < end + 5:
        raise ValueError("Invalid base support length")
    coords = tuple(tuple(data[start : start + 3]) for start in range(begin, end, 3))
    length = struct.unpack_from("<I", data, end)[0]
    stream_end = end + 4 + length
    if length < 1 or stream_end >= len(data):
        raise ValueError("Invalid arithmetic payload length")
    packet = FramePacket(
        posQ, coords, tuple(data[stream_end:]), data[end + 4 : stream_end], condition
    )
    packet._validate()
    return packet
