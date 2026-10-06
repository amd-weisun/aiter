# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Owned HIP IPC peer memory used by fused multi-GPU kernels."""

from __future__ import annotations

import torch

from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap as _Ipc

_HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11


def _alloc_base(dev_ptr: int) -> int:
    """The hipMalloc allocation that holds ``dev_ptr`` (torch sub-allocates), whose IPC
    handle is the one peers open."""
    import ctypes

    hip = _Ipc._load_hip()
    hip.hipPointerGetAttribute.restype = ctypes.c_int
    hip.hipPointerGetAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    base = ctypes.c_void_p()
    err = hip.hipPointerGetAttribute(
        ctypes.byref(base),
        ctypes.c_int(_HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR),
        ctypes.c_void_p(int(dev_ptr)),
    )
    _Ipc._hip_check(err, what="hipPointerGetAttribute")
    return int(base.value)


class SymmetricPeerBuffer:
    """Allocate one symmetric buffer and exchange its address with every rank.

    Remote HIP IPC mappings are closed by :meth:`close`. The local allocation is
    owned by ``storage`` and stays alive for the wrapper's lifetime.
    """

    def __init__(self, nbytes: int, rank: int = 0, npes: int = 1, group=None):
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        if not 0 <= rank < npes:
            raise ValueError(f"rank must be in [0, {npes}), got {rank}")
        device = torch.device("cuda", torch.cuda.current_device())
        self.rank = rank
        self.npes = npes
        self.group = group
        self.storage = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        self.local_address = self.storage.data_ptr()
        self._remote_bases: list[int] = []
        self._safety_barrier_complete = False

        if npes == 1:
            addresses = [self.local_address]
        else:
            import torch.distributed as dist

            base = _alloc_base(self.local_address)
            mine = (_Ipc.get_mem_handle_bytes(base), self.local_address - base)
            peers = [None] * npes
            dist.all_gather_object(peers, mine, group=group)
            addresses = []
            try:
                for peer_rank, (handle, offset) in enumerate(peers):
                    if peer_rank == rank:
                        addresses.append(self.local_address)
                    else:
                        remote_base = _Ipc.open_mem_handle(handle)
                        self._remote_bases.append(remote_base)
                        addresses.append(remote_base + offset)
            except Exception:
                for remote_base in self._remote_bases:
                    # best-effort cleanup: the original error is re-raised below
                    try:
                        _Ipc.close_mem_handle(remote_base)
                    except Exception:  # noqa: S110, BLE001
                        pass
                self._remote_bases.clear()
                raise
            dist.barrier(group=group)
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self) -> None:
        """Synchronize every rank, then close remote mappings.

        Every rank in the peer group must call this method in the same order.
        Repeated calls are safe. Failed closes remain tracked for a retry.
        """

        if not self._safety_barrier_complete:
            torch.cuda.synchronize(self.storage.device)
            if self.npes > 1:
                import torch.distributed as dist

                if not dist.is_initialized():
                    raise RuntimeError(
                        "the distributed process group must remain initialized until peer buffers close"
                    )
                dist.barrier(group=self.group)
            self._safety_barrier_complete = True
        failed = []
        first_error = None
        for base in self._remote_bases:
            try:
                _Ipc.close_mem_handle(base)
            # any failed close is kept for a retry
            except Exception as exc:  # noqa: BLE001
                failed.append(base)
                if first_error is None:
                    first_error = exc
        self._remote_bases = failed
        if first_error is not None:
            raise first_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
