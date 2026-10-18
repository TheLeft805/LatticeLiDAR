"""LSQ2: unknown-length, bounded, frame-at-a-time transport of GJA bodies.

Header 348 bytes; frame record 20 bytes plus body; end record 20 bytes.
Legacy LSQ1 remains in sequence_container. CRC covers metadata and payload.
The receiver must call finish() to distinguish a complete stream from EOF.
"""

import struct
import zlib
import hashlib

PREFIX = 328


def sha(data):
    return hashlib.sha256(data).hexdigest()


HEADER = struct.Struct("<4sHHII")
RECORD = struct.Struct("<BBHIQ")
U32 = struct.Struct("<I")
LEGACY = struct.Struct("<QI")


def ordered(previous, number, reset):
    if type(reset) is not bool or not 0 <= number <= 0xFFFFFFFF:
        raise ValueError("Invalid frame metadata")
    if previous is not None and number <= previous:
        raise ValueError("Frame order invalid")
    if (previous is None or number != previous + 1) and not reset:
        raise ValueError("Discontinuity requires reset")


class SequenceWriter:
    def __init__(self, sequence, prefix):
        if len(prefix) != PREFIX:
            raise ValueError("Invalid identity prefix")
        self.prefix = bytes(prefix)
        h = HEADER.pack(b"LSQ2", 2, 0, sequence, PREFIX)
        self.header = h + self.prefix + U32.pack(zlib.crc32(h + self.prefix))
        self.previous = None
        self.count = self.digest = 0
        self.closed = False

    def frame(self, number, reset, packet):
        if self.closed:
            raise ValueError("Stream already ended")
        ordered(self.previous, number, reset)
        if len(packet) < PREFIX + LEGACY.size or packet[:PREFIX] != self.prefix:
            raise ValueError("Packet identity differs")
        size, crc = LEGACY.unpack_from(packet, PREFIX)
        body = packet[PREFIX + LEGACY.size :]
        if size != len(body) or zlib.crc32(body) != crc:
            raise ValueError("Invalid original body")
        meta = RECORD.pack(1, int(reset), 0, number, size)
        record = meta + U32.pack(zlib.crc32(body, zlib.crc32(meta))) + body
        self.digest = zlib.crc32(record, self.digest)
        self.previous = number
        self.count += 1
        return record

    def finish(self):
        if self.closed or not self.count:
            raise ValueError("Empty or ended stream")
        meta = RECORD.pack(2, 0, 0, self.digest, self.count)
        self.closed = True
        return meta + U32.pack(zlib.crc32(meta))


class SequenceReader:
    def __init__(
        self, expected_prefix_sha, expected_sequence, max_body_bytes=64 * 1024 * 1024
    ):
        if max_body_bytes < 0:
            raise ValueError("Invalid body limit")
        self.expected_prefix_sha = expected_prefix_sha
        self.expected_sequence = expected_sequence
        self.limit = max_body_bytes
        self.buffer = bytearray()
        self.need = HEADER.size + PREFIX + U32.size
        self.prefix = None
        self.meta = None
        self.previous = None
        self.count = self.digest = 0
        self.ended = self.failed = False
        self.peak_buffer_bytes = 0

    def feed(self, chunk):
        """Yield each validated frame before consuming any subsequent record.

        Callers must consume this iterator. Buffer never exceeds one record,
        even when the supplied chunk contains an entire sequence.
        """
        if self.failed:
            raise ValueError("Reader failed")
        data = memoryview(chunk)
        try:
            while len(data):
                if self.ended:
                    raise ValueError("Trailing bytes")
                take = min(len(data), self.need - len(self.buffer))
                self.buffer.extend(data[:take])
                data = data[take:]
                self.peak_buffer_bytes = max(self.peak_buffer_bytes, len(self.buffer))
                if len(self.buffer) != self.need:
                    continue
                if self.prefix is None:
                    raw = bytes(self.buffer)
                    h = raw[: HEADER.size]
                    magic, version, flags, sequence, length = HEADER.unpack(h)
                    prefix = raw[HEADER.size : -4]
                    if (magic, version, flags, length) != (b"LSQ2", 2, 0, PREFIX):
                        raise ValueError("Invalid stream header")
                    if U32.unpack(raw[-4:])[0] != zlib.crc32(raw[:-4]):
                        raise ValueError("Header checksum differs")
                    if (
                        sequence != self.expected_sequence
                        or sha(prefix) != self.expected_prefix_sha
                    ):
                        raise ValueError("Stream identity differs")
                    self.prefix = prefix
                    self.buffer.clear()
                    self.need = RECORD.size + 4
                elif self.meta is None:
                    kind, reset, reserved, number, size = RECORD.unpack_from(
                        self.buffer
                    )
                    if reserved or kind not in (1, 2):
                        raise ValueError("Invalid record")
                    if kind == 2:
                        if (
                            reset
                            or size != self.count
                            or not size
                            or number != self.digest
                        ):
                            raise ValueError("End count/digest differs")
                        if U32.unpack_from(self.buffer, RECORD.size)[0] != zlib.crc32(
                            self.buffer[: RECORD.size]
                        ):
                            raise ValueError("End checksum differs")
                        self.buffer.clear()
                        self.ended = True
                    else:
                        if reset not in (0, 1) or size > self.limit:
                            raise ValueError("Invalid frame extent")
                        ordered(self.previous, number, bool(reset))
                        self.meta = (number, bool(reset), size)
                        self.need += size
                        # Complete zero-length records without waiting for another chunk.
                        if size == 0:
                            yield self._frame()
                else:
                    yield self._frame()
        except BaseException:
            self.failed = True
            raise

    def _frame(self):
        number, reset, size = self.meta
        meta = self.buffer[: RECORD.size]
        body = bytes(self.buffer[RECORD.size + 4 :])
        crc = zlib.crc32(body)
        if U32.unpack_from(self.buffer, RECORD.size)[0] != zlib.crc32(
            body, zlib.crc32(meta)
        ):
            raise ValueError("Frame checksum differs")
        self.digest = zlib.crc32(self.buffer, self.digest)
        packet = self.prefix + LEGACY.pack(size, crc) + body
        self.previous = number
        self.count += 1
        self.meta = None
        self.buffer.clear()
        self.need = RECORD.size + 4
        return number, reset, packet

    def finish(self):
        if self.failed or not self.ended or self.buffer or self.meta is not None:
            raise ValueError("Truncated or failed stream")
        return self.count
