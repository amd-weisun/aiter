#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""UT-level MoE sorting benchmark: OPUS vs FlyDSL.

Temporary script for PR review (prepare branch only). Sweeps token counts
1..16384 for production configs requested by reviewers (DeepSeek-V4, GPT-OSS).

Before timing each (config, T) point, runs all backends once and verifies
FlyDSL matches OPUS and CK (and OPUS matches CK).

Run from the aiter repo root on a GPU node, e.g. inside atom_bench:
  cd /path/to/aiter
  FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=0 \\
    python op_tests/bench_moe_sorting_opus_flydsl.py

Optional flags:
  --configs dsv4 gptoss   # subset of model configs (default: both)
  --skip-check            # skip correctness checks (timing only)
  --no-ck                 # skip CK timing column (correctness still uses CK unless --skip-check)
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import aiter
from aiter.ops.flydsl.kernels.moe_sorting_kernel import (
    moe_sorting_flydsl,
    moe_sorting_get_workspace_size,
)
from aiter.ops.flydsl.utils import is_flydsl_available

WARMUP = 10
BENCH = 100
GRAPH_REPLAY = 200

T_ALL = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

CONFIGS = {
    "gptoss": ("GPT-OSS", 40, 6, 128, 5120),
    "dsv4": ("DeepSeek-V4", 385, 7, 32, 7168),
}

SortingOutputs = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


def make_unique_ids(m: int, topk: int, num_experts: int, device: torch.device) -> torch.Tensor:
    ids = torch.zeros(m, topk, dtype=torch.int32, device=device)
    for t in range(m):
        perm = torch.randperm(num_experts, device=device)[:topk]
        ids[t] = perm.to(torch.int32)
    return ids


def alloc_outputs(
    m: int, topk: int, num_experts: int, unit_size: int, model_dim: int, device: torch.device
):
    max_padded = m * topk + num_experts * unit_size - topk
    max_blocks = (max_padded + unit_size - 1) // unit_size
    return (
        torch.empty(max_padded, dtype=torch.int32, device=device),
        torch.empty(max_padded, dtype=torch.float32, device=device),
        torch.empty(max_blocks, dtype=torch.int32, device=device),
        torch.empty(2, dtype=torch.int32, device=device),
        torch.empty((m, model_dim), dtype=torch.bfloat16, device=device),
    )


def _bench(run_fn, use_graph: bool) -> float:
    if use_graph:
        for _ in range(WARMUP):
            run_fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(device="cuda:0")
        with torch.cuda.stream(s):
            run_fn()
        with torch.cuda.graph(graph, stream=s):
            run_fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(GRAPH_REPLAY):
            graph.replay()
        en.record()
        torch.cuda.synchronize()
        return st.elapsed_time(en) / GRAPH_REPLAY * 1000

    for _ in range(WARMUP):
        run_fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(BENCH):
        run_fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / BENCH * 1000


def run_opus(ids, weights, num_experts, unit_size, m, topk, model_dim, device):
    si, sw, se, nv, mb = alloc_outputs(m, topk, num_experts, unit_size, model_dim, device)
    wsz = aiter.moe_sorting_opus_get_workspace_size(m, num_experts, topk, 0)
    ws = torch.empty(wsz, dtype=torch.uint8, device=device) if wsz > 0 else None
    aiter.moe_sorting_opus_fwd(
        ids, weights, si, sw, se, nv, mb, num_experts, unit_size, None, None, ws, 0
    )
    return si, sw, se, nv, mb


def run_flydsl(ids, weights, num_experts, unit_size, m, topk, model_dim, device):
    si, sw, se, nv, mb = alloc_outputs(m, topk, num_experts, unit_size, model_dim, device)
    wsz = moe_sorting_get_workspace_size(m, num_experts, topk, unit_size)
    ws = torch.empty(wsz, dtype=torch.int32, device=device) if wsz > 0 else None
    moe_sorting_flydsl(
        ids, weights, si, sw, se, nv, mb, num_experts, unit_size, None, None, ws
    )
    return si, sw, se, nv, mb


def run_ck(ids, weights, num_experts, unit_size, m, topk, model_dim, device):
    si, sw, se, nv, mb = alloc_outputs(m, topk, num_experts, unit_size, model_dim, device)
    aiter.moe_sorting_fwd(ids, weights, si, sw, se, nv, mb, num_experts, unit_size, None, None, 0)
    return si, sw, se, nv, mb


def bench_opus(ids, weights, num_experts, unit_size, m, topk, model_dim, device, use_graph):
    si, sw, se, nv, mb = alloc_outputs(m, topk, num_experts, unit_size, model_dim, device)
    wsz = aiter.moe_sorting_opus_get_workspace_size(m, num_experts, topk, 0)
    ws = torch.empty(wsz, dtype=torch.uint8, device=device) if wsz > 0 else None

    def _run():
        aiter.moe_sorting_opus_fwd(
            ids, weights, si, sw, se, nv, mb, num_experts, unit_size, None, None, ws, 0
        )

    return _bench(_run, use_graph)


def bench_flydsl(ids, weights, num_experts, unit_size, m, topk, model_dim, device, use_graph):
    si, sw, se, nv, mb = alloc_outputs(m, topk, num_experts, unit_size, model_dim, device)
    wsz = moe_sorting_get_workspace_size(m, num_experts, topk, unit_size)
    ws = torch.empty(wsz, dtype=torch.int32, device=device) if wsz > 0 else None

    def _run():
        moe_sorting_flydsl(
            ids, weights, si, sw, se, nv, mb, num_experts, unit_size, None, None, ws
        )

    return _bench(_run, use_graph)


def bench_ck(ids, weights, num_experts, unit_size, m, topk, model_dim, device, use_graph):
    si, sw, se, nv, mb = alloc_outputs(m, topk, num_experts, unit_size, model_dim, device)

    def _run():
        aiter.moe_sorting_fwd(
            ids, weights, si, sw, se, nv, mb, num_experts, unit_size, None, None, 0
        )

    return _bench(_run, use_graph)


def compare_sorting_outputs(
    ref: SortingOutputs,
    cand: SortingOutputs,
    *,
    topk: int,
    m: int,
    ref_name: str,
    cand_name: str,
) -> None:
    """Raise if candidate outputs differ from reference (same rules as op_tests/test_moe_sorting)."""
    ref_ids, ref_weights, ref_expert_ids, ref_nv, _ = ref
    cand_ids, cand_weights, cand_expert_ids, cand_nv, _ = cand

    mismatches: list[str] = []

    if not torch.equal(ref_nv, cand_nv):
        mismatches.append(
            f"num_valid_ids: {ref_name}={ref_nv.tolist()} vs {cand_name}={cand_nv.tolist()}"
        )

    num_tokens_post_pad = ref_nv[0].item()
    if not torch.equal(ref_ids[:num_tokens_post_pad], cand_ids[:num_tokens_post_pad]):
        mismatches.append("sorted_ids")

    pad_sentinel = (topk << 24) | m
    weight_mask = ref_ids != pad_sentinel
    if not torch.equal(ref_weights[weight_mask], cand_weights[weight_mask]):
        mismatches.append("sorted_weights")

    expert_mask = ref_expert_ids != -1
    if not torch.equal(ref_expert_ids[expert_mask], cand_expert_ids[expert_mask]):
        mismatches.append("sorted_expert_ids")

    if mismatches:
        raise RuntimeError(
            f"{cand_name} vs {ref_name} mismatch at M={m}, topk={topk}: {', '.join(mismatches)}"
        )


def verify_correctness(
    ids,
    weights,
    num_experts,
    unit_size,
    m,
    topk,
    model_dim,
    device,
) -> None:
    """Run OPUS, CK, FlyDSL once and verify all three agree."""
    opus_out = run_opus(ids, weights, num_experts, unit_size, m, topk, model_dim, device)
    ck_out = run_ck(ids, weights, num_experts, unit_size, m, topk, model_dim, device)
    flydsl_out = run_flydsl(ids, weights, num_experts, unit_size, m, topk, model_dim, device)

    compare_sorting_outputs(
        opus_out, ck_out, topk=topk, m=m, ref_name="OPUS", cand_name="CK"
    )
    compare_sorting_outputs(
        opus_out, flydsl_out, topk=topk, m=m, ref_name="OPUS", cand_name="FlyDSL"
    )
    compare_sorting_outputs(
        ck_out, flydsl_out, topk=topk, m=m, ref_name="CK", cand_name="FlyDSL"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=sorted(CONFIGS),
        default=sorted(CONFIGS),
        help="model configs to benchmark (default: all)",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="skip correctness checks and run timing only",
    )
    parser.add_argument("--no-ck", action="store_true", help="skip CK timing column")
    args = parser.parse_args()

    if not is_flydsl_available():
        print("FlyDSL is not available in this build; cannot benchmark FlyDSL path.", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        return 1

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("=" * 95)
    print("  MoE Sorting UT Benchmark — OPUS vs FlyDSL")
    print(f"  Decode (T<=256): CUDA graph, {GRAPH_REPLAY} replays")
    print(f"  Prefill (T>256): CUDA events, {BENCH} iterations")
    if args.skip_check:
        print("  Correctness: SKIPPED (--skip-check)")
    else:
        print("  Correctness: FlyDSL vs OPUS vs CK before each T")
    print("=" * 95)

    include_ck = not args.no_ck

    for key in args.configs:
        name, num_experts, topk, unit_size, model_dim = CONFIGS[key]
        print(f"\n{'─' * 95}")
        print(f"  {name}: E={num_experts}, topk={topk}, unit_size={unit_size}, model_dim={model_dim}")
        print(f"{'─' * 95}")
        header = f"  {'T':>6s} {'Path':>6s} {'OPUS':>8s} {'FlyDSL':>8s} {'vs OPUS':>9s}"
        if include_ck:
            header += f" {'CK':>8s}"
        if not args.skip_check:
            header += f" {'check':>6s}"
        print(header)
        sep = f"  {'─' * 6} {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 9}"
        if include_ck:
            sep += f" {'─' * 8}"
        if not args.skip_check:
            sep += f" {'─' * 6}"
        print(sep)

        n_ok = 0
        for m in T_ALL:
            torch.manual_seed(42)
            ids = make_unique_ids(m, topk, num_experts, device)
            weights = torch.rand(m, topk, device=device, dtype=torch.float32)
            use_graph = m <= 256
            path = "graph" if use_graph else "eager"

            check_status = "ok"
            if not args.skip_check:
                try:
                    verify_correctness(
                        ids, weights, num_experts, unit_size, m, topk, model_dim, device
                    )
                    n_ok += 1
                except Exception as exc:
                    check_status = "FAIL"
                    print(f"  correctness failed at T={m}: {exc}", file=sys.stderr)
                    return 1

            try:
                t_op = bench_opus(
                    ids, weights, num_experts, unit_size, m, topk, model_dim, device, use_graph
                )
            except Exception as exc:
                print(f"  OPUS failed at T={m}: {exc}", file=sys.stderr)
                t_op = None

            try:
                t_fl = bench_flydsl(
                    ids, weights, num_experts, unit_size, m, topk, model_dim, device, use_graph
                )
            except Exception as exc:
                print(f"  FlyDSL failed at T={m}: {exc}", file=sys.stderr)
                t_fl = None

            t_ck = None
            if include_ck:
                try:
                    t_ck = bench_ck(
                        ids, weights, num_experts, unit_size, m, topk, model_dim, device, use_graph
                    )
                except Exception:
                    t_ck = None

            op_s = f"{t_op:>8.1f}" if t_op is not None else f"{'N/A':>8s}"
            fl_s = f"{t_fl:>8.1f}" if t_fl is not None else f"{'N/A':>8s}"
            vs_op = (
                f"{(1 - t_fl / t_op) * 100:>+8.1f}%"
                if (t_op and t_fl)
                else f"{'N/A':>9s}"
            )
            row = f"  {m:>6d} {path:>6s} {op_s} {fl_s} {vs_op}"
            if include_ck:
                ck_s = f"{t_ck:>8.1f}" if t_ck is not None else f"{'N/A':>8s}"
                row += f" {ck_s}"
            if not args.skip_check:
                row += f" {check_status:>6s}"
            print(row)

        if not args.skip_check:
            print(f"  correctness: {n_ok}/{len(T_ALL)} passed")

    print(f"\n{'=' * 95}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
