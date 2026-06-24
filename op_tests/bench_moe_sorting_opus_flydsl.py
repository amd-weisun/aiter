#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""UT-level MoE sorting benchmark: OPUS vs FlyDSL.

Temporary script for PR review (prepare branch only). Sweeps token counts
1..16384 for production configs requested by reviewers (DeepSeek-V4, GPT-OSS).

Before timing each (config, T) point, runs OPUS and FlyDSL once. Each is checked
vs the CPU reference; FlyDSL is also checked directly vs OPUS.

Run from the aiter repo root on a GPU node, e.g. inside atom_bench:
  cd /path/to/aiter
  export PYTHONPATH=/workspace/aiter:$PYTHONPATH
  FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=0 \\
    python op_tests/bench_moe_sorting_opus_flydsl.py

Timing modes (--timing):
  micro  — direct kernel calls; CUDA graph (T<=256) or events (T>256) [default]
  kineto — moe_sorting() dispatch + PyTorch profiler (quiet by default)

Optional flags:
  --configs dsv4 gptoss   # subset of model configs (default: both)
  --skip-check            # skip correctness checks (timing only)
  --kineto-detail         # one-line per-kernel breakdown per (backend, T)
  --kineto-log            # full aiter-style profiler table (very verbose)
  --kineto-iters N        # profiler iterations (default: 21)
  --kineto-warmup N       # warmup before profile (default: 2)
  --tokens T [T ...]      # subset of T values (default: full sweep)
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import warnings
from contextlib import contextmanager

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import pandas as pd
import torch
import torch.profiler as tpf

import aiter
import aiter.fused_moe as fm
from aiter import dtypes
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.test_common import checkAllclose, post_process_data, run_iters, run_iters_rotate
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

BLOCK_SIZE_M = 32
SORT_KERNEL_RE = re.compile(
    r"moe.?sort|sorting|opus|flydsl|p0v2|p0_scatter|p1_count|p23|count_kernel|"
    r"decode|scatter|multiphase|clear.?ws|clear_workspace",
    re.IGNORECASE,
)

NativeOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
SortingOutputs = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


def set_moe_sorting_backend(backend: str) -> None:
    """Mirror op_tests/test_moe_sorting.py backend selection."""
    if backend == "flydsl":
        if not is_flydsl_available():
            raise RuntimeError("backend=flydsl requested but FlyDSL is not available")
        fm._USE_CK_MOE_SORTING = False
        fm._USE_OPUS_MOE_SORTING = False
    elif backend == "opus":
        fm._USE_CK_MOE_SORTING = False
        fm._USE_OPUS_MOE_SORTING = True
    else:
        raise ValueError(f"unknown backend for bench: {backend}")


def moe_sorting_native(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int = BLOCK_SIZE_M,
    expert_mask=None,
    num_local_tokens=None,
) -> NativeOutputs:
    """PyTorch reference implementation (mirrors op_tests/test_moe_sorting.py)."""
    device = topk_ids.device
    m, topk = topk_ids.shape
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    init_val = topk << 24 | m
    sorted_ids = torch.full(
        (max_num_tokens_padded,), init_val, dtype=dtypes.i32, device=device
    )
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=dtypes.i32, device=device
    )
    num_tokens_post_pad = torch.empty((2), dtype=dtypes.i32, device=device)

    if num_local_tokens is not None:
        topk_ids = topk_ids[: num_local_tokens.item()]

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    skip_expert_num = 0
    for expert_id in range(num_experts):
        if expert_mask is not None and expert_mask[expert_id] == 0:
            skip_expert_num += 1
            continue
        token_id, topk_id = torch.where(topk_ids == expert_id)
        tokens_num = token_id.numel()
        sorted_expert_ids_num = (tokens_num + block_size - 1) // block_size
        tokens_num_pad = sorted_expert_ids_num * block_size
        sorted_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = (
            topk_id << 24 | token_id
        )
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = topk_weights[
            token_id, topk_id
        ]
        sorted_ids_begin = sorted_ids_begin + tokens_num_pad
        sorted_expert_ids[
            sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num
        ] = (expert_id - skip_expert_num)
        sorted_expert_ids_begin = sorted_expert_ids_begin + sorted_expert_ids_num

    num_tokens_post_pad[0] = sorted_ids_begin
    num_tokens_post_pad[1] = topk_ids.shape[0]

    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad


def make_sorting_inputs(
    m: int,
    num_experts: int,
    topk: int,
    model_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Same input path as op_tests/test_moe_sorting.py (fused_topk on random scores)."""
    inp = torch.randn((m, model_dim), dtype=dtype, device=device)
    score = torch.rand((m, num_experts), device=device, dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    return topk_ids, topk_weights


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


def get_trace_perf_table(prof, num_iters: int) -> tuple[float, pd.DataFrame]:
    """Same aggregation as aiter.test_common.get_trace_perf, but returns the table too."""
    assert num_iters > 1
    warm_iter = 1
    num_iters -= warm_iter
    rows = []
    cols = [
        "name",
        "self_cpu_time_total",
        "self_device_time_total",
        "device_type",
        "device_index",
    ]
    for el in prof.events():
        rows.append([getattr(el, x, None) for x in cols])
    df = pd.DataFrame(rows, columns=cols)
    dropped_indexs, dropped_num = post_process_data(df, num_iters + warm_iter, warm_iter)
    df = df.drop(dropped_indexs)
    iter_init = 0
    df["cnt"] = 1
    rets = []
    for name, d in df.groupby("name", sort=False):
        kernel_num_per_iter = iter_init
        if str(d["device_type"].iat[0]).split(".")[-1] != "CUDA":
            kernel_num_per_iter = 1
        r = d.iloc[kernel_num_per_iter:][
            ["cnt", "self_cpu_time_total", "self_device_time_total"]
        ].sum()
        if not r.empty:
            device_type = str(d["device_type"].iat[0]).split(".")[-1]
            r["name"] = name
            r["device_type"] = device_type
            r["device_index"] = str(d["device_index"].iat[0])
            if device_type == "CUDA":
                r["device_time_sum"] = r["self_device_time_total"]
                r["host_time_sum"] = 0
            else:
                r["host_time_sum"] = r["self_device_time_total"]
                r["device_time_sum"] = 0
            r["device_time_avg"] = (
                r["device_time_sum"] / r["cnt"] if r["cnt"] > 0 else 0
            )
        rets.append(r)
    df = pd.DataFrame(rets)
    out_cols = [
        "name",
        "cnt",
        "host_time_sum",
        "device_time_sum",
        "device_time_avg",
        "device_type",
        "device_index",
    ]
    out_cols = [el for el in out_cols if el in df.columns]
    df = df[(df.host_time_sum > 0) | (df.device_time_sum > 0)]
    df = df[out_cols].sort_values(["host_time_sum", "device_time_sum"], ignore_index=True)
    actual_iters = num_iters + warm_iter - dropped_num
    avg_name = "[avg us/iter]"
    for el in ["host_time_sum", "device_time_sum"]:
        if el == "host_time_sum":
            df.at[avg_name, el] = df[el].sum() / num_iters
        else:
            df.at[avg_name, el] = df[el].sum() / actual_iters
    total_us = float(df.at[avg_name, "device_time_sum"])
    return total_us, df


def _print_kineto_sort_kernels(df: pd.DataFrame, backend: str, m: int) -> None:
    cuda = df[(df["device_type"] == "CUDA") & (df["name"] != "[avg us/iter]")]
    hits = cuda[cuda["name"].astype(str).str.contains(SORT_KERNEL_RE, regex=True)]
    if hits.empty:
        hits = cuda
    print(f"    kineto kernels ({backend}, T={m}):")
    for _, row in hits.iterrows():
        name = str(row["name"])
        if len(name) > 72:
            name = name[:69] + "..."
        print(f"      {row['device_time_avg']:8.1f} us  {name}")


def _print_kineto_full_table(df: pd.DataFrame, backend: str, m: int) -> None:
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_colwidth", 72)
    pd.set_option("display.float_format", "{:,.1f}".format)
    print(f"\n    kineto table ({backend}, T={m}):\n{df.to_string()}\n")


@contextmanager
def _suppress_aiter_log_more():
    """Keep post_process_data quiet unless user opted into --kineto-log."""
    prev = os.environ.get("AITER_LOG_MORE")
    os.environ["AITER_LOG_MORE"] = "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("AITER_LOG_MORE", None)
        else:
            os.environ["AITER_LOG_MORE"] = prev


def bench_kineto_moe_sorting(
    backend: str,
    topk_ids,
    topk_weights,
    num_experts: int,
    model_dim: int,
    dtype: torch.dtype,
    unit_size: int,
    *,
    num_iters: int,
    num_warmup: int,
    show_detail: bool,
    show_log: bool,
) -> float:
    """Profile moe_sorting() via kineto (aiter run_perftest aggregation)."""
    set_moe_sorting_backend(backend)
    args = (
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        dtype,
        unit_size,
        None,
        None,
        0,
    )
    rotate_args = [(copy.deepcopy(args), {}) for _ in range(num_iters - 1)] + [(args, {})]
    run_iters(num_warmup, moe_sorting, *args)
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Profiler clears events at the end of each cycle.*",
            category=UserWarning,
        )
        with tpf.profile(
            activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
            profile_memory=False,
            with_stack=False,
            with_modules=True,
        ) as prof:
            run_iters_rotate(num_iters, moe_sorting, rotate_args)
            torch.cuda.synchronize()
        with _suppress_aiter_log_more():
            total_us, df = get_trace_perf_table(prof, num_iters)
    if show_log:
        _print_kineto_full_table(df, backend, topk_ids.shape[0])
    elif show_detail:
        _print_kineto_sort_kernels(df, backend, topk_ids.shape[0])
    return total_us


def _bench_micro(run_fn, use_graph: bool) -> float:
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


def bench_opus_micro(ids, weights, num_experts, unit_size, m, topk, model_dim, device, use_graph):
    si, sw, se, nv, mb = alloc_outputs(m, topk, num_experts, unit_size, model_dim, device)
    wsz = aiter.moe_sorting_opus_get_workspace_size(m, num_experts, topk, 0)
    ws = torch.empty(wsz, dtype=torch.uint8, device=device) if wsz > 0 else None

    def _run():
        aiter.moe_sorting_opus_fwd(
            ids, weights, si, sw, se, nv, mb, num_experts, unit_size, None, None, ws, 0
        )

    return _bench_micro(_run, use_graph)


def bench_flydsl_micro(ids, weights, num_experts, unit_size, m, topk, model_dim, device, use_graph):
    si, sw, se, nv, mb = alloc_outputs(m, topk, num_experts, unit_size, model_dim, device)
    wsz = moe_sorting_get_workspace_size(m, num_experts, topk, unit_size)
    ws = torch.empty(wsz, dtype=torch.int32, device=device) if wsz > 0 else None

    def _run():
        moe_sorting_flydsl(
            ids, weights, si, sw, se, nv, mb, num_experts, unit_size, None, None, ws
        )

    return _bench_micro(_run, use_graph)


def verify_against_native(
    native_out: NativeOutputs,
    gpu_out: SortingOutputs,
    *,
    topk: int,
    m: int,
    backend_name: str,
) -> None:
    ref_ids, ref_weights, ref_expert_ids, ref_nv = native_out
    gpu_ids, gpu_weights, gpu_expert_ids, gpu_nv, _ = gpu_out

    errs: dict[str, float] = {}
    errs["num_tokens_post_padded"] = checkAllclose(
        ref_nv, gpu_nv, atol=0, msg=f"{backend_name} num_tokens_post_padded", printLog=False
    )
    weight_mask = ref_ids != (topk << 24 | m)
    num_tokens_post_pad = ref_nv[0].item()
    errs["sorted_ids"] = checkAllclose(
        ref_ids[:num_tokens_post_pad],
        gpu_ids[:num_tokens_post_pad],
        msg=f"{backend_name} sorted_ids",
        printLog=False,
    )
    errs["sorted_weights"] = checkAllclose(
        ref_weights[weight_mask],
        gpu_weights[weight_mask],
        msg=f"{backend_name} sorted_weights",
        printLog=False,
    )
    expert_mask = ref_expert_ids != -1
    errs["sorted_expert_ids"] = checkAllclose(
        ref_expert_ids[expert_mask],
        gpu_expert_ids[expert_mask],
        msg=f"{backend_name} sorted_expert_ids",
        printLog=False,
    )
    bad = {k: v for k, v in errs.items() if v}
    if bad:
        raise RuntimeError(
            f"{backend_name} mismatch vs CPU reference at M={m}, topk={topk}: "
            f"mismatch fractions {bad}"
        )


def verify_gpu_pair(
    ref_out: SortingOutputs,
    cand_out: SortingOutputs,
    native_out: NativeOutputs,
    *,
    topk: int,
    m: int,
    ref_name: str,
    cand_name: str,
) -> None:
    native_ids, native_weights, native_expert_ids, native_nv = native_out
    ref_ids, ref_weights, ref_expert_ids, ref_nv, _ = ref_out
    cand_ids, cand_weights, cand_expert_ids, cand_nv, _ = cand_out

    errs: dict[str, float] = {}
    errs["num_valid_ids"] = checkAllclose(
        ref_nv, cand_nv, atol=0, msg=f"{cand_name} vs {ref_name} num_valid_ids", printLog=False
    )
    weight_mask = native_ids != (topk << 24 | m)
    num_tokens_post_pad = native_nv[0].item()
    errs["sorted_ids"] = checkAllclose(
        ref_ids[:num_tokens_post_pad],
        cand_ids[:num_tokens_post_pad],
        msg=f"{cand_name} vs {ref_name} sorted_ids",
        printLog=False,
    )
    errs["sorted_weights"] = checkAllclose(
        ref_weights[weight_mask],
        cand_weights[weight_mask],
        msg=f"{cand_name} vs {ref_name} sorted_weights",
        printLog=False,
    )
    expert_mask = native_expert_ids != -1
    errs["sorted_expert_ids"] = checkAllclose(
        ref_expert_ids[expert_mask],
        cand_expert_ids[expert_mask],
        msg=f"{cand_name} vs {ref_name} sorted_expert_ids",
        printLog=False,
    )
    bad = {k: v for k, v in errs.items() if v}
    if bad:
        raise RuntimeError(
            f"{cand_name} vs {ref_name} mismatch at M={m}, topk={topk}: "
            f"mismatch fractions {bad}"
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
    native_out = moe_sorting_native(ids, weights, num_experts, unit_size)
    opus_out = run_opus(ids, weights, num_experts, unit_size, m, topk, model_dim, device)
    flydsl_out = run_flydsl(ids, weights, num_experts, unit_size, m, topk, model_dim, device)

    verify_against_native(native_out, opus_out, topk=topk, m=m, backend_name="OPUS")
    verify_against_native(native_out, flydsl_out, topk=topk, m=m, backend_name="FlyDSL")
    verify_gpu_pair(
        opus_out, flydsl_out, native_out, topk=topk, m=m, ref_name="OPUS", cand_name="FlyDSL"
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
        "--timing",
        choices=["micro", "kineto"],
        default="micro",
        help="micro=direct kernel CUDA graph/events; kineto=moe_sorting()+profiler",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="skip correctness checks and run timing only",
    )
    parser.add_argument(
        "--kineto-detail",
        action="store_true",
        help="print compact per-kernel device_time_avg (kineto mode only)",
    )
    parser.add_argument(
        "--kineto-log",
        action="store_true",
        help="print full profiler table per (backend, T); very verbose",
    )
    parser.add_argument(
        "--kineto-iters",
        type=int,
        default=21,
        help="profiler iterations (default: 21; use 101 to match run_perftest)",
    )
    parser.add_argument("--kineto-warmup", type=int, default=2, help="warmup before profile")
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        metavar="T",
        help=f"token counts to run (default: all {T_ALL})",
    )
    args = parser.parse_args()

    if args.tokens is not None:
        token_list = []
        for t in args.tokens:
            if t not in T_ALL:
                print(f"unknown T={t}; allowed: {T_ALL}", file=sys.stderr)
                return 1
            token_list.append(t)
        args.tokens = token_list
    else:
        args.tokens = T_ALL

    if args.kineto_iters < 2:
        print("--kineto-iters must be >= 2", file=sys.stderr)
        return 1

    if not is_flydsl_available():
        print("FlyDSL is not available in this build; cannot benchmark FlyDSL path.", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        return 1

    if args.timing == "kineto" and args.kineto_log:
        os.environ["AITER_LOG_MORE"] = "1"

    dtype = dtypes.d_dtypes["bf16"]
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("=" * 80)
    print("  MoE Sorting UT Benchmark — OPUS vs FlyDSL")
    if args.timing == "micro":
        print(f"  Timing: micro — graph T<=256 ({GRAPH_REPLAY} replays), events T>256 ({BENCH} iters)")
    else:
        print(
            f"  Timing: kineto — moe_sorting() + profiler "
            f"({args.kineto_warmup} warmup, {args.kineto_iters} iters, quiet output)"
        )
    if args.skip_check:
        print("  Correctness: SKIPPED (--skip-check)")
    else:
        print("  Correctness: OPUS + FlyDSL vs CPU; FlyDSL vs OPUS (direct kernel fwd)")
    print("=" * 80)

    for key in args.configs:
        name, num_experts, topk, unit_size, model_dim = CONFIGS[key]
        print(f"\n{'─' * 80}")
        print(f"  {name}: E={num_experts}, topk={topk}, unit_size={unit_size}, model_dim={model_dim}")
        print(f"{'─' * 80}")
        header = f"  {'T':>6s} {'Path':>6s} {'OPUS':>8s} {'FlyDSL':>8s} {'vs OPUS':>9s}"
        if not args.skip_check:
            header += f" {'check':>6s}"
        print(header)
        sep = f"  {'─' * 6} {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 9}"
        if not args.skip_check:
            sep += f" {'─' * 6}"
        print(sep)

        n_ok = 0
        for m in args.tokens:
            torch.manual_seed(42)
            ids, weights = make_sorting_inputs(m, num_experts, topk, model_dim, dtype, device)
            use_graph = args.timing == "micro" and m <= 256
            path = "graph" if use_graph else ("kineto" if args.timing == "kineto" else "eager")

            check_status = "ok"
            if not args.skip_check:
                try:
                    verify_correctness(
                        ids, weights, num_experts, unit_size, m, topk, model_dim, device
                    )
                    n_ok += 1
                except Exception as exc:
                    print(f"  correctness failed at T={m}: {exc}", file=sys.stderr)
                    return 1

            try:
                if args.timing == "kineto":
                    t_op = bench_kineto_moe_sorting(
                        "opus",
                        ids,
                        weights,
                        num_experts,
                        model_dim,
                        dtype,
                        unit_size,
                        num_iters=args.kineto_iters,
                        num_warmup=args.kineto_warmup,
                        show_detail=args.kineto_detail,
                        show_log=args.kineto_log,
                    )
                else:
                    t_op = bench_opus_micro(
                        ids, weights, num_experts, unit_size, m, topk, model_dim, device, use_graph
                    )
            except Exception as exc:
                print(f"  OPUS failed at T={m}: {exc}", file=sys.stderr)
                t_op = None

            try:
                if args.timing == "kineto":
                    t_fl = bench_kineto_moe_sorting(
                        "flydsl",
                        ids,
                        weights,
                        num_experts,
                        model_dim,
                        dtype,
                        unit_size,
                        num_iters=args.kineto_iters,
                        num_warmup=args.kineto_warmup,
                        show_detail=args.kineto_detail,
                        show_log=args.kineto_log,
                    )
                else:
                    t_fl = bench_flydsl_micro(
                        ids, weights, num_experts, unit_size, m, topk, model_dim, device, use_graph
                    )
            except Exception as exc:
                print(f"  FlyDSL failed at T={m}: {exc}", file=sys.stderr)
                t_fl = None

            op_s = f"{t_op:>8.1f}" if t_op is not None else f"{'N/A':>8s}"
            fl_s = f"{t_fl:>8.1f}" if t_fl is not None else f"{'N/A':>8s}"
            vs_op = (
                f"{(1 - t_fl / t_op) * 100:>+8.1f}%"
                if (t_op and t_fl)
                else f"{'N/A':>9s}"
            )
            row = f"  {m:>6d} {path:>6s} {op_s} {fl_s} {vs_op}"
            if not args.skip_check:
                row += f" {check_status:>6s}"
            print(row)

        if not args.skip_check:
            print(f"  correctness: {n_ok}/{len(T_ALL)} passed")

    print(f"\n{'=' * 80}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
