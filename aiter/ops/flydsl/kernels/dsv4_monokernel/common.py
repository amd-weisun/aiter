# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.


"""The FlyDSL MonoKernel shared pieces the DeepSeek-V4 MonoKernel uses.

Vendored verbatim from FlyDSL (ROCm/FlyDSL, commit 709bfcc): ``kernels/monokernel/``
``config.py`` (the MoE mode / format contract and two constants), ``layout.py``
(cache-modifier and sentinel constants), ``formats.py`` (MX formats), ``packing.py``
(FP8 / BF16 tile packing), ``reference.py`` (golden helpers) and ``ops.py`` (device
helpers). aiter does not carry FlyDSL's ``kernels/monokernel`` package, so these
are kept together here; buffer and DPP helpers come from aiter's own copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32

# ---- from FlyDSL kernels/monokernel/config.py


class MoeMode(str, Enum):
    """Public arithmetic modes for the expert up/gate and down projections."""

    W8A8 = "w8a8"
    W8A16 = "w8a16"
    A16W4 = "a16w4"
    A8W4 = "a8w4"


class ExpertActivation(str, Enum):
    """Activation representation consumed by both expert projections."""

    FP8_BLOCK128 = "fp8_block128"
    MXFP8_BLOCK32 = "mxfp8_block32"
    BF16 = "bf16"


class ExpertWeight(str, Enum):
    """Packed expert-weight representation."""

    FP8_BLOCK128 = "fp8_block128"
    MXFP4_BLOCK32 = "mxfp4_block32"


@dataclass(frozen=True)
class MoeFormat:
    activation: ExpertActivation
    weight: ExpertWeight

    @property
    def activation_group(self) -> int | None:
        if self.activation is ExpertActivation.FP8_BLOCK128:
            return 128
        if self.activation is ExpertActivation.MXFP8_BLOCK32:
            return 32
        return None


MOE_FORMATS = {
    MoeMode.W8A8: MoeFormat(ExpertActivation.FP8_BLOCK128, ExpertWeight.FP8_BLOCK128),
    MoeMode.W8A16: MoeFormat(ExpertActivation.BF16, ExpertWeight.FP8_BLOCK128),
    MoeMode.A16W4: MoeFormat(ExpertActivation.BF16, ExpertWeight.MXFP4_BLOCK32),
    MoeMode.A8W4: MoeFormat(ExpertActivation.MXFP8_BLOCK32, ExpertWeight.MXFP4_BLOCK32),
}


def as_moe_mode(value: MoeMode | str) -> MoeMode:
    """Normalize a public mode argument and report supported values clearly."""

    if isinstance(value, MoeMode):
        return value
    try:
        return MoeMode(value)
    except ValueError as error:
        choices = ", ".join(mode.value for mode in MoeMode)
        raise ValueError(
            f"unsupported MoE mode {value!r}; expected one of: {choices}"
        ) from error


def moe_format(value: MoeMode | str) -> MoeFormat:
    """Return the independent activation and weight formats for a public mode."""

    return MOE_FORMATS[as_moe_mode(value)]


# ---- from FlyDSL kernels/monokernel/config.py


SCALE_BM = 128


FP8_MAX = 448.0


# ---- from FlyDSL kernels/monokernel/layout.py


NEG = -1.0e30


CM_DEV = 16


CM_SYS = 17


# ---- from FlyDSL kernels/monokernel/formats.py


def float_to_e8m0(x: torch.Tensor) -> torch.Tensor:
    """Round positive FP32 values to E8M0 exponent bytes."""

    bits = x.float().contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF).to(torch.uint8)
    round_up = ((bits & 0x400000) != 0) & (
        ((bits & 0x200000) != 0) | ((bits & 0x1FFFFF) != 0) | (exponent != 0)
    )
    exponent = exponent + round_up.to(torch.uint8)
    return torch.where(exponent == 0xFF, torch.full_like(exponent, 0xFF), exponent)


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 exponent bytes to FP32 power-of-two scales."""

    exponent = scale.view(torch.uint8)
    bits = exponent.to(torch.int32) << 23
    bits = torch.where(exponent == 0, torch.full_like(bits, 0x00400000), bits)
    bits = torch.where(exponent == 0xFF, torch.full_like(bits, 0x7F800001), bits)
    return bits.view(torch.float32)


def _float_to_fp4_codes(x: torch.Tensor) -> torch.Tensor:
    """Round finite FP32 values to E2M1 codes, with ties to even."""

    magnitude = x.float().abs().clamp(max=6.0)
    boundaries = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=x.device)
    code = torch.bucketize(magnitude, boundaries, right=False).to(torch.uint8)
    # At these midpoints the upper code has an even mantissa bit.
    code = code + ((magnitude == 0.75) | (magnitude == 1.75) | (magnitude == 3.5)).to(
        torch.uint8
    )
    return code | ((x < 0).to(torch.uint8) << 3)


def quantize_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., K]`` to packed MXFP4 plus row-major per-1x32 E8M0 scales."""

    if w.shape[-1] % 32:
        raise ValueError(
            f"MXFP4 K dimension must be divisible by 32, got {w.shape[-1]}"
        )
    shape = w.shape
    blocks = w.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = blocks.abs().amax(dim=-1)
    scale = float_to_e8m0(amax / 4.0)
    scale_f32 = e8m0_to_float(scale).clamp_min(torch.finfo(torch.float32).tiny)
    codes = _float_to_fp4_codes(blocks / scale_f32.unsqueeze(-1)).reshape(shape)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    return packed, scale.contiguous()


def dequantize_mxfp4(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode row-major packed MXFP4 values and per-1x32 E8M0 scales."""

    q = q.view(torch.uint8)
    codes = q.repeat_interleave(2, dim=-1)
    codes[..., 0::2] &= 0xF
    codes[..., 1::2] >>= 4
    values = torch.tensor(
        (
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ),
        dtype=torch.float32,
        device=q.device,
    )
    return values[codes.long()] * e8m0_to_float(scale).repeat_interleave(32, dim=-1)


def quant_dequant_mxfp8(x: torch.Tensor) -> torch.Tensor:
    """Per-1x32 MXFP8 E4M3 quantization, returned dequantized in FP32."""

    if x.shape[-1] % 32:
        raise ValueError(
            f"MXFP8 K dimension must be divisible by 32, got {x.shape[-1]}"
        )
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = blocks.abs().amax(dim=-1)
    scale = float_to_e8m0(amax / 448.0)
    scale_f32 = e8m0_to_float(scale).clamp_min(torch.finfo(torch.float32).tiny)
    q = (
        (blocks / scale_f32.unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .float()
    )
    return (q * scale_f32.unsqueeze(-1)).reshape(shape)


# ---- from FlyDSL kernels/monokernel/packing.py


def pack_fp8(q: torch.Tensor) -> torch.Tensor:
    """Pack FP8 ``[..., N, K]`` for the kernel's 16-row, 64-K MFMA tiles."""

    *lead, rows, k = q.shape
    if rows % 16 or k % 64:
        raise ValueError(
            f"FP8 matrix dimensions must be divisible by (16, 64), got {(rows, k)}"
        )
    w8 = q.view(torch.uint8).reshape(*lead, rows // 16, 16, k // 64, 2, 4, 8)
    nlead = len(lead)
    order = list(range(nlead)) + [nlead + position for position in (0, 2, 4, 1, 3, 5)]
    return w8.permute(*order).contiguous().view(-1)


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """Pack BF16 ``[N, K]`` for the kernel's 16-row, 64-K MFMA tiles."""

    if w.ndim != 2:
        raise ValueError(f"BF16 packing expects a matrix, got shape {tuple(w.shape)}")
    rows, k = w.shape
    if rows % 16 or k % 64:
        raise ValueError(
            f"BF16 matrix dimensions must be divisible by (16, 64), got {(rows, k)}"
        )
    w16 = w.view(torch.int16).reshape(rows // 16, 16, k // 64, 2, 4, 8)
    return w16.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1)


# ---- from FlyDSL kernels/monokernel/reference.py


def scale_shape(rows: int, k: int, bk: int):
    return ((rows + SCALE_BM - 1) // SCALE_BM, k // bk)


def _rand_fp8(rows, k, bk, gen, device, lead=()):
    q = (torch.randn(*lead, rows, k, generator=gen, device=device) * 16).clamp(
        -FP8_MAX, FP8_MAX
    )
    q = q.to(torch.float8_e4m3fn)
    sr, sk = scale_shape(rows, k, bk)
    s = (torch.rand(*lead, sr, sk, generator=gen, device=device) * 0.4 + 0.8) / (
        16 * k**0.5
    )
    return q, s


def dequant(
    q: torch.Tensor, s: torch.Tensor, bk: int, bm: int = SCALE_BM
) -> torch.Tensor:
    rows, _k = q.shape
    sf = s.repeat_interleave(bm, 0)[:rows].repeat_interleave(bk, 1)
    return q.float() * sf


def dequant_expert(
    q: torch.Tensor, scale: torch.Tensor, weight: ExpertWeight
) -> torch.Tensor:
    """Decode one logical expert matrix for the torch reference."""

    if weight is ExpertWeight.MXFP4_BLOCK32:
        return dequantize_mxfp4(q, scale)
    return dequant(q, scale, 128)


def bf(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 and back (the precision of MFMA activation operands)."""
    return x.to(torch.bfloat16).float()


def quant_dequant(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Per-``block`` dynamic FP8 E4M3FN quantization of the last dim, returned dequantized."""
    xb = x.float().reshape(*x.shape[:-1], -1, block)
    amax = xb.abs().amax(-1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (xb / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(x.shape)


# ---- from FlyDSL kernels/monokernel/ops.py


def rsrc(addr):
    return bo.create_buffer_resource_from_addr(addr)


def uniform(value):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(value).ir_value()))


def uniform_f32(value):
    return uniform(fx.Float32(value).bitcast(fx.Int32)).bitcast(fx.Float32)


def write_lane_i32(value, lane, vector):
    """Write one i32 into ``lane`` of a wave-distributed value."""

    return fx.Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.writelane.i32",
            [as_ir_value(fx.Int32(item)) for item in (value, lane, vector)],
            [],
            [],
        )
    )


def mem_realtime():
    """Read the device-wide 64-bit realtime counter."""

    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def _hardware_f32(name, value):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(value).ir_value()], [], [])
    )


def rsq(value):
    return _hardware_f32("llvm.amdgcn.rsq.f32", value)


def rcp(value):
    return _hardware_f32("llvm.amdgcn.rcp.f32", value)


def exp(value):
    return _hardware_f32("llvm.amdgcn.exp2.f32", fx.Float32(value) * 1.4426950408889634)


def xshfl(value, offset):
    """Return the value from ``lane ^ offset`` using VALU/DPP operations."""

    if offset >= 16:
        return value.shuffle_xor(offset, 64)
    is_float = isinstance(value, fx.Float32)
    source = value.bitcast(fx.Int32) if is_float else fx.Int32(value)
    if offset == 8:
        result = fx.Int32(update_dpp_i32(source, source, 0x118, 0xF, 0xC, False))
        result = fx.Int32(update_dpp_i32(result, source, 0x108, 0xF, 0x3, False))
    elif offset == 4:
        result = fx.Int32(update_dpp_i32(source, source, 0x114, 0xF, 0xA, False))
        result = fx.Int32(update_dpp_i32(result, source, 0x104, 0xF, 0x5, False))
    elif offset == 2:
        result = fx.Int32(update_dpp_i32(source, source, 0x4E, 0xF, 0xF, False))
    else:
        result = fx.Int32(update_dpp_i32(source, source, 0xB1, 0xF, 0xF, False))
    return result.bitcast(fx.Float32) if is_float else result


def wave_umax(value):
    return fx.Int32(fx.coop.warp_reduce(fx.Uint32(value), fx.ReductionOp.MAX, width=64))


def xred(value, offset, op):
    """Combine a value with ``lane ^ offset`` using a symmetric operation."""

    if offset < 16:
        return op(value, xshfl(value, offset))
    is_float = isinstance(value, fx.Float32)
    source = as_ir_value(value.bitcast(fx.Int32) if is_float else fx.Int32(value))
    swap = rocdl.permlane32_swap if offset == 32 else rocdl.permlane16_swap
    pair = swap(
        llvm.StructType.get_literal([T.i32, T.i32]), source, source, False, False
    )
    lhs, rhs = (fx.Int32(llvm.extractvalue(T.i32, pair, [index])) for index in range(2))
    if is_float:
        return op(lhs.bitcast(fx.Float32), rhs.bitcast(fx.Float32))
    return op(type(value)(lhs), type(value)(rhs))


def fp8_roundtrip(lhs, rhs):
    """Round an f32 pair through E4M3FN and return the f32 pair."""

    word = rocdl.cvt_pk_fp8_f32(T.i32, lhs, rhs, fx.Int32(0), False)
    pair_type = fx.Vector.make_type(2, fx.Float32)
    pair = fx.Vector(rocdl.cvt_pk_f32_fp8(res=pair_type, src=word, word_sel=False))
    return pair[0], pair[1]


def f8_word(k):
    """Map an FP8 activation index to the packed LDS word order."""

    return (k // 64) * 16 + ((k % 32) // 8) * 4 + ((k % 64) // 32) * 2 + (k % 8) // 4


def fp8_to_bf16x8(word0, word1):
    """Convert two dwords of eight FP8 values to a BF16 vector."""

    one = as_ir_value(fx.Float32(1.0))
    parts = []
    for word in (word0, word1):
        for half in range_constexpr(2):
            pair = fx.Vector(
                rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16), as_ir_value(word), one, bool(half)
                )
            )
            parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def mxfp8_to_bf16x8(word0, word1, scale):
    """Convert two dwords of eight FP8 values with one E8M0 block scale to BF16."""

    scale = as_ir_value(scale)
    parts = []
    for word in (word0, word1):
        for half in range_constexpr(2):
            pair = fx.Vector(
                rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16), as_ir_value(word), scale, bool(half)
                )
            )
            parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def mxfp4_to_bf16x8(word, scale):
    """Convert one packed dword of eight scaled E2M1 values to BF16."""

    parts = []
    for select in range_constexpr(4):
        pair = fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp4(
                T.vec(2, T.bf16),
                as_ir_value(word),
                as_ir_value(scale),
                select,
            )
        )
        parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)
