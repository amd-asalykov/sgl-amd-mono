# SPDX-License-Identifier: Apache-2.0
"""Time a target verify step (and an MTP draft step) of the launches, replayed from
CUDA graphs on the ``harness.py`` fixtures.

  python test/kernels/qwen38_flash_next/bench_launches.py --R 1,4,8 --L 4 [--layers 48]

``--layers 48`` times the full model's 13 launches (loads all 48 layers)."""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness as H  # noqa: E402
from check_launches import _ints  # noqa: E402

from sgl_amd_mono.qwen38_flash_next.gfx950.config import HC_DIM  # noqa: E402
from sgl_amd_mono.qwen38_flash_next.gfx950.runner import (  # noqa: E402
    DraftRunner,
    TargetRunner,
)


def _time(fn, iters: int) -> float:
    """Median us of a captured ``fn`` over ``iters`` replays."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    times = []
    for _ in range(iters):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return sorted(times)[len(times) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=_ints, default=[1, 4, 8])
    ap.add_argument("--L", type=_ints, default=[4])
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--draft", action="store_true")
    args = ap.parse_args()
    variants = {"plugin": None}
    for extra in os.environ.get("BENCH_EXTRA", "").split(","):
        if extra:  # a kernel module of gfx950/ to time too, e.g. a previous step's
            import importlib

            mod = importlib.import_module(
                f"sgl_amd_mono.qwen38_flash_next.gfx950.{extra}"
            )
            variants[extra] = mod.build_qwen38_launch
    if args.draft:
        weights = H.draft_weights()
    else:
        weights = H.target_weights(args.layers)
    for L in args.L:
        if args.draft:
            pools = H.Pools(("qsa",), L)
            hidden = torch.randn(8 * L, HC_DIM, device=H.DEV).bfloat16()
        else:
            pools = H.Pools(weights.kinds, L)
        cos = H.cos_sin(max(pools.contexts) + L + 64)
        for R in args.R:
            row = []
            for name, build in variants.items():
                if args.draft:
                    run = DraftRunner(weights, L, H.DEV, build=build)
                    run.set_deadline(30.0)
                    ext, dec = (
                        H.step_batch(pools, R, L, seed=R),
                        H.step_batch(pools, R, 1, seed=R),
                    )
                    us = _time(
                        lambda: run.forward(
                            ext, hidden[: R * L], pools, pools.req_to_token, cos
                        ),
                        args.iters,
                    )
                    ud = _time(
                        lambda: run.forward(
                            dec, hidden[:R], pools, pools.req_to_token, cos
                        ),
                        args.iters,
                    )
                    row.append(f"{name} extend {us:.1f} us decode {ud:.1f} us")
                else:
                    run = TargetRunner(weights, L, H.DEV, build=build)
                    run.set_deadline(30.0)
                    b = H.step_batch(pools, R, L, seed=R)
                    us = _time(lambda: run.forward(b, pools, pools, cos), args.iters)
                    row.append(f"{name} {us:.1f} us")
            what = "draft" if args.draft else f"target ({args.layers} layers)"
            print(f"{what} R={R} L={L}: " + "; ".join(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
