"""FP32 frozen readout kernels; unknown slots retain their learned columns."""

import triton
import triton.language as tl


@triton.jit
def pack_inputs(
    Features,
    Bare,
    Own,
    Sensor,
    Tokens,
    Mask,
    History,
    HistoryMask,
    Offsets,
    Beam,
    Available,
    NormW,
    NormB,
    Base,
    Rest,
    TokenCopy,
    MaskCopy,
    BareCopy,
    AS: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, 32)
    f = tl.load(Features + row * 32 + c)
    mean = tl.sum(f, 0) / 32.0
    z = f - mean
    var = tl.sum(z * z, 0) / 32.0
    norm = z * (1.0 / tl.sqrt(var + EPS)) * tl.load(NormW + c) + tl.load(NormB + c)
    tl.store(Base + row * 70 + c, norm)
    d = tl.arange(0, 16)
    logits = tl.load(Bare + row * 16 + d)
    centered = logits - tl.max(logits, 0)
    logp = centered - tl.log(tl.sum(tl.exp(centered), 0))
    tl.store(Base + row * 70 + 32 + d, logp)
    tl.store(BareCopy + row * 16 + d, logits)
    own = tl.load(Own + row)
    cat = tl.arange(0, 32)
    tl.store(Base + row * 70 + 48 + cat, (cat == own).to(tl.float32), cat < 17)
    tl.store(
        Base + row * 70 + 65 + cat, tl.load(Sensor + row * 5 + cat, cat < 5, 0), cat < 5
    )
    j = tl.arange(0, 256)
    node = j // 9
    bit = j % 9
    value = tl.load(History + row * 27 + node, j < 243, 0).to(tl.int32)
    mask = tl.load(HistoryMask + row * 27 + node, j < 243, 0)
    item = tl.where(bit == 8, mask, ((value >> bit) & 1)).to(tl.float32)
    tl.store(Rest + row * 292 + j, item, j < 243)
    k = tl.arange(0, 32)
    tl.store(
        Rest + row * 292 + 243 + k,
        tl.load(Offsets + row * 24 + k, k < 24, 0).to(tl.float32) / 3.0,
        k < 24,
    )
    tl.store(
        Rest + row * 292 + 267 + k, tl.load(Beam + row * 21 + k, k < 21, 0), k < 21
    )
    tl.store(
        Rest + row * 292 + 288 + k,
        tl.load(Available + row * AS + k, k < 4, 0).to(tl.float32),
        k < 4,
    )
    tl.store(
        TokenCopy + row * 30 + k, tl.load(Tokens + row * 30 + k, k < 30, 16), k < 30
    )
    tl.store(MaskCopy + row * 30 + k, tl.load(Mask + row * 30 + k, k < 30, 0), k < 30)


@triton.jit
def slot_projection(
    Continuous,
    Tokens,
    Mask,
    Weight,
    Out,
    N: tl.constexpr,
    START: tl.constexpr,
    COUNT: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, 128)
    acc = tl.load(Continuous + row * 128 + c).to(tl.float32)
    for j in range(START, START + COUNT):
        token = tl.load(Tokens + row * 30 + j).to(tl.int32)
        flag = tl.load(Mask + row * 30 + j).to(tl.float32)
        acc = acc + tl.load(Weight + (j * 18 + token) * 128 + c)
        acc = acc + flag * tl.load(Weight + (j * 18 + 17) * 128 + c)
    tl.store(Out + row * 128 + c, acc)


@triton.jit
def norm_film(
    H,
    Residual,
    E,
    GammaBeta,
    W,
    B,
    Out,
    HNext,
    N: tl.constexpr,
    EPS: tl.constexpr,
    ADD: tl.constexpr,
    FILM: tl.constexpr,
    HalfOut,
    DUAL: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, 128)
    h = tl.load(H + row * 128 + c).to(tl.float32)
    if ADD:
        h = h + tl.load(Residual + row * 128 + c).to(tl.float32)
    tl.store(HNext + row * 128 + c, h)
    mean = tl.sum(h, 0) / 128.0
    z = h - mean
    var = tl.sum(z * z, 0) / 128.0
    value = z * (1.0 / tl.sqrt(var + EPS)) * tl.load(W + c) + tl.load(B + c)
    if FILM:
        value = (
            value * (1.0 + tl.load(GammaBeta + c))
            + tl.load(GammaBeta + 128 + c)
            + tl.load(E + row * 128 + c).to(tl.float32)
        )
    tl.store(Out + row * 128 + c, value)
    if DUAL:
        tl.store(HalfOut + row * 128 + c, value)
