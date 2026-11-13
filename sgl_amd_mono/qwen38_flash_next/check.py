# SPDX-License-Identifier: Apache-2.0
"""SGLANG_AMD_MONO_CHECK=1: at build, the weights the launches read of three layers
against the checkpoint's bytes (``weights``); then (eager steps only) every routed
verify step also runs SGLang's own forward of the step first, on the caches as they
were, and the launches' outputs are checked against it: the logits and the hidden
states (the residual streams after the last layer, which the MTP draft reads), each
against its ``spec.NUMERICS`` bound, on the worst row. Every check is logged; a miss
is logged as a violation.

The 13 launches run the whole model, so the check covers the step, not a layer.
Both forwards write the same cache rows (the draft tokens' KV, index keys and
verify scratch) from the same committed state, and the launches write them last,
so the step leaves the caches as an unchecked step would.

At TP > 1 every rank runs the check in lockstep: the reference is SGLang's TP
forward (its collectives on every rank), every rank compares its own outputs (the
logits are all-gathered, so the full vocab), rank 0 logs the step, and the
launches' residual streams, which every rank computes, must be bitwise equal
across the ranks. The weights check compares each rank's shards with the
checkpoint's matching slices (notes/tp_sglang_sharding.md)."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/runtime_context.py": ("get_server_args",),
    "srt/arg_groups/fields/*.py": ("model_path", "revision"),
    "srt/layers/logits_processor.py": ("next_token_logits", "hidden_states"),
    "srt/model_executor/forward_batch_info.py": (
        "positions",
        "batch_size",
        "req_pool_indices",
    ),
    "srt/model_executor/forward_context.py": (
        "get_req_to_token_pool",
        "get_token_to_kv_pool",
    ),
    "srt/mem_cache/memory_pool.py": ("get_mamba_indices", "conv", "temporal"),
}


def _committed_state(kinds: list[str], forward_batch) -> list[tuple]:
    """The committed recurrent state the launches read: SGLang's verify conv update
    rolls the draft tokens into the GDN (and PLE short-conv) state in place."""
    from sglang.srt.model_executor.forward_context import (
        get_req_to_token_pool,
        get_token_to_kv_pool,
    )

    from sgl_amd_mono.qwen38_flash_next.gfx950.runner import gdn_cache, ple_states

    rtp, kvp = get_req_to_token_pool(), get_token_to_kv_pool()
    slots = rtp.get_mamba_indices(forward_batch.req_pool_indices).long()
    saved = []
    for layer_id, kind in enumerate(kinds):
        if kind != "gdn":
            continue
        cache = gdn_cache(kvp, layer_id)
        for t in (cache.conv[0], cache.temporal):
            saved.append((t, slots, t[slots].clone()))
    ple = ple_states(kvp, 1)[0]
    saved.append((ple, None, ple.clone()))
    return saved


def reference(lm, input_ids, positions, forward_batch, kinds: list[str]) -> dict:
    """SGLang's forward of the step: its logits and hidden states. The committed
    state it advances is restored, so the launches read the state as it was."""
    saved = _committed_state(kinds, forward_batch)
    out = type(lm).forward(lm, input_ids, positions, forward_batch)
    for t, idx, v in saved:
        if idx is None:
            t.copy_(v)
        else:
            t[idx] = v
    return {
        "logits": out.next_token_logits.clone(),
        "hidden": out.hidden_states.clone(),
    }


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """The worst row's cosine and relative difference."""
    a, b = a.float().flatten(1), b.float().flatten(1)
    rel = (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)
    return {
        "cosine": F.cosine_similarity(a, b, dim=1).min().item(),
        "rel": rel.max().item(),
        "max_abs": (a - b).abs().max().item(),
    }


def _cross_rank_equal(t: torch.Tensor, group) -> bool:
    """Whether ``t`` holds the same bits on every rank of the CPU group ``group``."""
    import torch.distributed as dist

    mine = t.detach().contiguous().view(torch.uint8).flatten().cpu()
    got = [torch.empty_like(mine) for _ in range(dist.get_world_size(group))]
    dist.all_gather(got, mine, group=group)
    return all(torch.equal(g, got[0]) for g in got)


def compare(ref: dict, out, forward_batch, runner=None) -> None:
    from sgl_amd_mono.qwen38_flash_next.spec import NUMERICS

    tp = getattr(runner, "tp", 1)
    rank = getattr(runner, "rank", 0)
    if tp > 1 and not _cross_rank_equal(out.hidden_states, runner.cpu_group):
        logger.error(
            "monokernel check violation: rank %d's residual streams differ from "
            "another rank's (the replicated streams must be bitwise equal)",
            rank,
        )

    got = {"logits": out.next_token_logits, "hidden": out.hidden_states}
    values = {
        f"verify.{name}.{metric}": v
        for name in ("logits", "hidden")
        for metric, v in _metrics(got[name], ref[name]).items()
    }
    pos = forward_batch.positions
    pos = pos[0] if pos.dim() == 2 else pos
    if rank == 0:
        logger.info(
            "monokernel check step (%d requests, positions %d..%d): %s",
            int(forward_batch.batch_size),
            pos.min().item(),
            pos.max().item(),
            " ".join(f"{k}={v:.6g}" for k, v in values.items()),
        )
    for name, bound in NUMERICS.bounds.items():
        v = values.get(name)
        if v is None:
            continue
        ok = v >= bound if name.endswith(".cosine") else v <= bound
        if not ok:
            logger.error("monokernel check violation: rank %d %s = %.6g, bound %g", rank, name, v, bound)  # fmt: skip


# ---------------------------------------------------------------- weights
# the checkpoint's names: the text model's prefix, and the tensors each FP8
# projection the kernel reads merges (kernel name -> checkpoint names in the layer)
PREFIX = "model.language_model."
CHECKED_LAYERS = {0: "gdn", 1: "gdn", 3: "qsa"}  # a GDN, the PLE and a QSA layer
SHARED_EXPERT = 512  # fused after the 512 routed experts
CHECKED_EXPERTS = (0, 255, 511, SHARED_EXPERT)
# keyed by layer kind: both kinds name their output projection "out"
PROJ_SOURCES = {
    "gdn": {
        "ip": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
        "out": ("linear_attn.out_proj",),
        "ba": ("linear_attn.in_proj_b", "linear_attn.in_proj_a"),
    },
    "qsa": {
        "qp": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        "qp2": ("self_attn.indexer.index_qk_proj",),
        "out": ("self_attn.o_proj",),
    },
}


def _checkpoint_dir() -> str:
    """The served checkpoint's directory: only its safetensors are read, so a
    partial HF cache resolves."""
    import os

    from sglang.srt.runtime_context import get_server_args

    args = get_server_args()
    path = args.model_path
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            path,
            revision=args.revision,
            local_files_only=True,
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        )
    return path


def _expert_name(e: int) -> str:
    return "mlp.shared_expert" if e == SHARED_EXPERT else f"mlp.experts.{e}"


def _load(path: str, names: list[str], device) -> dict:
    import json
    import os

    from safetensors import safe_open

    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    by_file: dict = {}
    for n in names:
        by_file.setdefault(index[n], []).append(n)
    out = {}
    for fname, ns in by_file.items():
        with safe_open(os.path.join(path, fname), "pt", device="cpu") as f:
            for n in ns:
                out[n] = f.get_tensor(n).to(device)
    return out


def _gdn_channels(tp: int, r: int, device) -> torch.Tensor:
    """Rank r's [q_r | k_r | v_r] rows of the GDN [q | k | v] channels (in_proj_qkv,
    conv1d), each sub-block split separately."""
    from sgl_amd_mono.qwen38_flash_next.gfx950.config import GDN_KEY_DIM, GDN_VAL_DIM

    kd, vd = GDN_KEY_DIM // tp, GDN_VAL_DIM // tp
    a = lambda s, n: torch.arange(s, s + n, device=device)  # noqa: E731
    return torch.cat([a(r * kd, kd), a(GDN_KEY_DIM + r * kd, kd), a(2 * GDN_KEY_DIM + r * vd, vd)])  # fmt: skip


def _small_shard(name: str, t: torch.Tensor, tp: int, r: int) -> torch.Tensor:
    """Rank r's slice of a checkpoint small tensor (replicated ones whole)."""
    if tp == 1:
        return t
    if name.endswith("linear_attn.conv1d.weight"):
        return t[_gdn_channels(tp, r, t.device)]
    if name.endswith(("linear_attn.A_log", "linear_attn.dt_bias")):
        n = t.shape[0] // tp
        return t[r * n : (r + 1) * n]
    return t


def _proj_shard(kind: str, key: str, j: int, w, s, tp: int, r: int):
    """Rank r's slice of source ``j`` of a projection (weight [N, K], scale [N]):
    column-parallel sources by rows, the row-parallel output projection by K."""
    if tp == 1 or key == "qp2":  # the QSA indexer is replicated
        return w, s
    if key == "out":
        k = w.shape[1] // tp
        return w[:, r * k : (r + 1) * k], s
    n = w.shape[0] // tp
    if kind == "gdn" and key == "ip" and j == 0:  # in_proj_qkv: [q_r | k_r | v_r]
        idx = _gdn_channels(tp, r, w.device)
        return w[idx], s[idx]
    if kind == "qsa" and j > 0:  # k_proj / v_proj: KV head r * 2 // tp (TP 4: pairs)
        h = r * 2 // tp
        return w[h * 256 : (h + 1) * 256], s[h * 256 : (h + 1) * 256]
    return w[r * n : (r + 1) * n], s[r * n : (r + 1) * n]


def _expert_shard(want: list, tp: int, r: int) -> list:
    """Rank r's intermediate chunk of an expert's (gate, up, down, gate_s, up_s,
    down_s) checkpoint tensors: gate / up rows, down columns (2 FP4 a byte, 32 a
    scale)."""
    if tp == 1:
        return want
    g, u, d, gs, us, ds = want
    i = g.shape[0] // tp
    rows = lambda t: t[r * i : (r + 1) * i]  # noqa: E731
    cols = lambda t, per: t[:, r * i // per : (r + 1) * i // per]  # noqa: E731
    return [rows(g), rows(u), cols(d, 2), rows(gs), rows(us), cols(ds, 32)]


def weights(model, kinds: list[str], tp: int = 1, rank: int = 0) -> None:
    """Check mode, at build: the tensors the launches read of layers 0, 1 and 3
    (SGLang's params, its AITER-shuffled experts and FP8 projections un-shuffled)
    against the checkpoint's bytes (at TP > 1, the rank's slices of them). A layout
    SGLang changed would not fault, it would produce plausible garbage."""
    from sgl_amd_mono.qwen38_flash_next.gfx950 import packing
    from sgl_amd_mono.qwen38_flash_next.gfx950 import weights as views

    path, layers = _checkpoint_dir(), model.model.layers
    for i, kind in CHECKED_LAYERS.items():
        assert kinds[i] == kind, (i, kinds[i])
        layer, p = layers[i], f"{PREFIX}layers.{i}."
        small = views.checkpoint_dict(layer, kind, layer.ple is not None, tp)
        table = views.gdn_proj(tp) if kind == "gdn" else views.qsa_proj(tp)
        proj = views.proj_params(layer, table)
        names = [p + k for k in small]
        names += [f"{p}{x}.{y}" for k in proj for x in PROJ_SOURCES[kind][k] for y in ("weight", "weight_scale")]  # fmt: skip
        for e in CHECKED_EXPERTS:
            src = _expert_name(e)
            names += [f"{p}{src}.{w}_proj.{y}" for w in ("gate", "up", "down") for y in ("weight", "weight_scale")]  # fmt: skip
        ref = _load(path, names, layer.mlp.experts.w13_weight.device)
        bad = [
            k
            for k, t in small.items()
            if not torch.equal(
                t.float().reshape(-1),
                _small_shard(k, ref[p + k], tp, rank).float().reshape(-1),
            )
        ]
        for k, (w, sc) in proj.items():
            srcs = PROJ_SOURCES[kind][k]
            parts = [
                _proj_shard(
                    kind,
                    k,
                    j,
                    ref[f"{p}{x}.weight"].view(torch.uint8),
                    ref[f"{p}{x}.weight_scale"].float().reshape(-1),
                    tp,
                    rank,
                )
                for j, x in enumerate(srcs)
            ]
            ref_w = torch.cat([a for a, _ in parts])
            ref_s = torch.cat([b for _, b in parts])
            n = ref_w.shape[0]
            got = packing.unshuffle16(w)
            if views._get(layer, table[k][0] + ".weight_scale") is None:
                # re-quantized from SGLang's dequantized bf16 (views.requantize_fp8_rows):
                # the codes exact, the scales within bf16's rounding
                scale_ok = bool(
                    ((sc[:n] - ref_s).abs() <= 2.0**-8 * ref_s.abs()).all()
                )
            else:
                scale_ok = torch.equal(sc[:n], ref_s)
            if not (torch.equal(got[:n], ref_w) and scale_ok and not got[n:].any()):
                bad.append(k)
        moe = views.moe_params(layer.mlp.experts, tp)
        for e in CHECKED_EXPERTS:
            src = _expert_name(e)
            got = packing.unshuffle_moe_expert(moe, e, tp)
            want = [
                ref[f"{p}{src}.{w}_proj.{y}"].view(torch.uint8)
                for y in ("weight", "weight_scale")
                for w in ("gate", "up", "down")
            ]
            want = _expert_shard(want, tp, rank)
            bad += [f"expert {e} {n}" for n, g, r in zip(("gate", "up", "down", "gate_s", "up_s", "down_s"), got, want) if not torch.equal(g, r)]  # fmt: skip
        del ref, proj
        if bad:
            logger.error(
                "monokernel check violation: layer %d reads tensors that differ from "
                "the checkpoint: %s",
                i,
                ", ".join(bad),
            )
        else:
            logger.info(
                "monokernel check weights layer %d (TP %d rank %d): %d tensors, %d "
                "projections and %d experts equal the checkpoint",
                i,
                tp,
                rank,
                len(small),
                len(table),
                len(CHECKED_EXPERTS),
            )
