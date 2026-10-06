# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Public API for the DeepSeek-V4 decode MonoKernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiter.ops.flydsl.kernels.dsv4_monokernel.common import MoeMode

if TYPE_CHECKING:
    from aiter.ops.flydsl.kernels.dsv4_monokernel.op import Dsv4MonoKernel

__all__ = ["Dsv4MonoKernel", "MoeMode"]


def __getattr__(name: str):
    """Load the GPU wrapper only when callers request it."""

    if name == "Dsv4MonoKernel":
        from aiter.ops.flydsl.kernels.dsv4_monokernel.op import Dsv4MonoKernel

        return Dsv4MonoKernel
    raise AttributeError(name)
