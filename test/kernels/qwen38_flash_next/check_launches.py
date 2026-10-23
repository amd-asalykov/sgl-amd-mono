# SPDX-License-Identifier: Apache-2.0
"""Run the Qwen3.8-Flash-Next launches on one gfx950 GPU (``harness.py``): every
output and every pool tensor compared bitwise with a reference, and every step
checked to write only its own rows (its KV slots and their compressed groups, its
requests' ring, RoPE and scratch rows; a pad row only the dummy slot 0; no
committed state).

The reference is the launches' own first eager run: a second eager run and every
CUDA-graph replay must equal it.

  python test/kernels/qwen38_flash_next/check_launches.py --what target --R 1-8 --L 2,4
  python test/kernels/qwen38_flash_next/check_launches.py --what draft --R 1-8 --L 2,4
  ... --graph: also capture the steps in CUDA graphs at several widths and replay
  them out of order against the reference's eager outputs.

Prints one line per (what, R, L, mode) and ``ALL BITWISE`` or the differing
tensors with their max |diff|; exits 1 on any difference, stray row or error word."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness as H  # noqa: E402

from sgl_amd_mono.qwen38_flash_next.gfx950.config import HC_DIM  # noqa: E402
from sgl_amd_mono.qwen38_flash_next.gfx950.runner import (  # noqa: E402
    DraftRunner,
    TargetRunner,
)


def _ints(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


# ------------------------------------------------------------------ precompile
def _compile(job) -> float:
    _, args, kwargs = job
    import inspect

    from flydsl.utils.env import CompileEnvManager

    from sgl_amd_mono.qwen38_flash_next.gfx950.kernel import build_qwen38_launch

    build = build_qwen38_launch
    t = time.time()
    launch = build(*args, **kwargs)
    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    var = CompileEnvManager.options["compile_only"].env_var
    os.environ[var] = "1"
    try:
        launch(*([0] * len(names)), stream=torch.cuda.current_stream())
    finally:
        os.environ.pop(var, None)
    return time.time() - t


def precompile(jobs, procs: int) -> None:
    jobs = list(jobs)
    if not jobs:
        return
    t = time.time()
    with ProcessPoolExecutor(procs, mp_context=get_context("spawn")) as ex:
        list(ex.map(_compile, jobs))
    print(f"precompiled {len(jobs)} kernels in {time.time() - t:.0f} s", flush=True)


def _freeze(key):
    args, kwargs = key
    return args, tuple(sorted(kwargs.items()))


def _jobs(runners, keys_of):
    jobs = []
    for which, runner in runners:
        for args, kwargs in keys_of(runner):
            jobs.append((which, args, tuple(sorted(kwargs.items()))))
    return [(w, a, dict(k)) for w, a, k in dict.fromkeys(jobs)]


# ------------------------------------------------------------------ compare
class Report:
    def __init__(self):
        self.bad = 0

    def compare(self, label: str, got: dict, ref: dict, errors=None, snap=None) -> None:
        """Bitwise per tensor; ``snap``: also name the pool tensors the step wrote
        (a vacuous match would write none)."""
        worst = []
        for name, r in ref.items():
            same, d = H.diff(got[name], r)
            if not same:
                worst.append(f"{name} {d:.3g}")
        if errors and any(errors):
            worst.append(f"error words {errors}")
        x = ref["x"].float()
        if not torch.isfinite(x).all() or x.abs().max() == 0:
            worst.append("x is zero or not finite")
        self.bad += bool(worst)
        wrote = ""
        if snap is not None:
            names = [n for n, t in snap.items() if not torch.equal(t, ref[n])]
            wrote = f" (wrote {len(names)} pool tensors: {_short(names)}; |x| max {x.abs().max():.3g})"
        print(
            f"{label}: "
            + ("ALL BITWISE" if not worst else "DIFF " + ", ".join(worst))
            + wrote,
            flush=True,
        )


def check_rows(rep: Report, label: str, got: dict, snap: dict, allowed: dict) -> None:
    """The rows the step wrote outside its own (another request's, a pad row's
    past the dummy slot, or a committed state) fail the run."""
    bad = H.stray_rows(got, snap, allowed)
    rep.bad += bool(bad)
    print(
        f"{label} rows: " + ("ONLY ITS OWN" if not bad else "STRAY " + "; ".join(bad)),
        flush=True,
    )


def _short(names):
    kinds = sorted({n.split(".")[-1] for n in names})
    return ",".join(kinds)


def _run(runner_call, pools, snap):
    pools.restore(snap)
    x, h = runner_call()
    torch.cuda.synchronize()
    out = {"x": x.clone(), "h": h.clone()} | {
        k: t.clone() for k, t in pools.tensors().items()
    }
    return out


# ------------------------------------------------------------------ target
def check_target(args, rep: Report) -> None:
    weights = H.target_weights(args.layers)
    for L in args.L:
        pools = H.Pools(weights.kinds, L)
        cos = H.cos_sin(max(pools.contexts) + L + 64)
        new = TargetRunner(weights, L, H.DEV)
        new.set_deadline(30.0)
        ref = new  # the reference: the launches' own first eager run
        batches = {
            R: H.step_batch(pools, R, L, seed=R, mrope=args.mrope) for R in args.R
        }
        if 3 in batches:
            batches["3+pad"] = H.pad_batch(batches[3], L)
        precompile(
            _jobs(
                [("new", new)],
                lambda run: [
                    k
                    for R in batches
                    for k in run.launch_keys(batches[R], pools, pools, cos)
                ],
            ),
            args.procs,
        )
        snap = pools.snapshot()
        refs = {}
        for R, b in batches.items():
            refs[R] = _run(lambda: ref.forward(b, pools, pools, cos), pools, snap)
            got = _run(lambda: new.forward(b, pools, pools, cos), pools, snap)
            if R == "3+pad":
                # the pad requests' rows are unspecified (all pad rows write the dummy
                # slots at once, as in SGLang); every real request's must be bitwise
                rows = 3 * L
                got |= {"x": got["x"][:rows], "h": got["h"][:rows]}
                refs[R] = refs[R] | {"x": refs[R]["x"][:rows], "h": refs[R]["h"][:rows]}
                for name, idx in H.pad_rows(pools, b, 3).items():
                    if idx is not None:
                        got[name][idx] = refs[R][name][idx]
            rep.compare(
                f"target R={R} L={L} eager", got, refs[R], new.error_words(), snap
            )
            check_rows(rep, f"target R={R} L={L}", got, snap, H.allowed_rows(pools, b))
        if args.graph:
            _target_graphs(weights, L, pools, cos, batches, refs, snap, rep)
        pools.restore(snap)
        pools = ref = new = None  # free the GPU memory before the next L
        torch.cuda.empty_cache()


def _target_graphs(weights, L, pools, cos, batches, refs, snap, rep):
    runner = TargetRunner(weights, L, H.DEV)
    runner.set_deadline(30.0)
    widths = [R for R in (1, 3, 8, 5) if R in batches and R in refs]
    graphs, outs = {}, {}
    for R in widths:
        b = batches[R]
        pools.restore(snap)
        runner.forward(b, pools, pools, cos)  # warm: compile, tables
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs[R] = runner.forward(b, pools, pools, cos)
        graphs[R] = g
    for R in widths[::-1] + widths:
        pools.restore(snap)
        graphs[R].replay()
        torch.cuda.synchronize()
        x, h = outs[R]
        got = {"x": x.clone(), "h": h.clone()} | {
            k: t.clone() for k, t in pools.tensors().items()
        }
        rep.compare(f"target R={R} L={L} graph", got, refs[R], runner.error_words())


# ------------------------------------------------------------------ draft
def check_draft(args, rep: Report) -> None:
    weights = H.draft_weights()
    for L in args.L:
        pools = H.Pools(("qsa",), L)
        cos = H.cos_sin(max(pools.contexts) + L + 64)
        new = DraftRunner(weights, L, H.DEV)
        new.set_deadline(30.0)
        ref = new  # the reference: the launch's own first eager run
        g = torch.Generator(device=H.DEV).manual_seed(11)
        hidden = (torch.randn(8 * L, HC_DIM, generator=g, device=H.DEV)).bfloat16()
        cases = {}
        for R in args.R:
            cases[("extend", R)] = (H.step_batch(pools, R, L, seed=R), hidden[: R * L])
            cases[("decode", R)] = (H.step_batch(pools, R, 1, seed=100 + R), hidden[:R])

        def keys(run):
            out = []
            for b, _ in cases.values():
                meta = {"rtt_stride": pools.req_to_token.stride(0), "pos_stride": 0}
                out.append(
                    run.launch_key(
                        b.batch_size, b.input_ids.numel() // b.batch_size, meta
                    )
                )
            return out

        precompile(_jobs([("new", new)], keys), args.procs)
        snap = pools.snapshot()
        refs = {}
        for (mode, R), (b, hs) in cases.items():
            call = lambda run: (
                lambda: run.forward(b, hs, pools, pools.req_to_token, cos)
            )  # noqa: E731
            refs[(mode, R)] = _run(call(ref), pools, snap)
            got = _run(call(new), pools, snap)
            label = f"draft {mode} R={R} L={L if mode == 'extend' else 1}"
            rep.compare(f"{label} eager", got, refs[(mode, R)], new.error_words(), snap)
            check_rows(rep, label, got, snap, H.allowed_rows(pools, b))
        if args.graph:
            runner = DraftRunner(weights, L, H.DEV)
            runner.set_deadline(30.0)
            graphs, outs = {}, {}
            keys_ = [k for k in cases if k[1] in (1, 3, 8)]
            for k in keys_:
                b, hs = cases[k]
                pools.restore(snap)
                runner.forward(b, hs, pools, pools.req_to_token, cos)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    outs[k] = runner.forward(b, hs, pools, pools.req_to_token, cos)
                graphs[k] = gr
            for k in keys_[::-1] + keys_:
                pools.restore(snap)
                graphs[k].replay()
                torch.cuda.synchronize()
                x, h = outs[k]
                got = {"x": x.clone(), "h": h.clone()} | {
                    n: t.clone() for n, t in pools.tensors().items()
                }
                rep.compare(
                    f"draft {k[0]} R={k[1]} L={L if k[0] == 'extend' else 1} graph",
                    got,
                    refs[k],
                    runner.error_words(),
                )
        pools.restore(snap)
        pools = None  # free the GPU memory before the next L
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="target,draft")
    ap.add_argument("--R", type=_ints, default=_ints("1-8"))
    ap.add_argument("--L", type=_ints, default=[2, 4])
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--mrope", action="store_true", help="[3, rows] positions")
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    rep = Report()
    torch.manual_seed(0)
    for what in args.what.split(","):
        {"target": check_target, "draft": check_draft}[what](args, rep)
    print("RESULT:", "PASS" if rep.bad == 0 else f"FAIL ({rep.bad})", flush=True)
    return int(rep.bad > 0)


if __name__ == "__main__":
    sys.exit(main())
