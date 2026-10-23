# SPDX-License-Identifier: Apache-2.0
"""``SGLANG_AMD_MONO_CHECK=1``: an eager routed step against SGLang's own forward.

SGLang's forward runs first on the step's inputs; the regions both write are
kept, put back as they were, and the launches run. Each output (the next-token
logits, the hidden streams, and every pool region the step writes) is then
compared, and a value past its bound in the spec's ``NUMERICS`` is logged as a
``monokernel check violation``. Float regions report, over the rows SGLang's forward
wrote, ``rel`` (the error's L2 norm over the reference's) and ``cosine`` (of all of
them), and ``row_cosine`` (the least row's, not bounded); integer regions ``max_abs``. Each layer's region counts toward its kind's value
(``kv``, ``ring``, ``ckv``, ``gdn_ssm``, ...), the worst one; layers 0-3 also as
``first_<kind>``.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

PLE_LAYER = 1  # the PLE block's layer (0-based)
# the first launch block's layers (0-3) also report on their own, as first_<kind>:
# near the embedding, before 48 layers compound the numerics, their bounds are tight
FIRST_BLOCK = 4
RESTORE_ONLY = "committed."  # regions put back after SGLang's forward, not compared
# slot 0 is the dummy slot: pad rows (and SGLang's non-boundary compressions) write it
DUMMY_ROW_REGIONS = ("kv", "ckv")


def _qsa_regions(out: dict, i, kvp, loc) -> None:
    out[f"kv.{i}.k"] = (kvp.get_key_buffer(i), loc)
    out[f"kv.{i}.v"] = (kvp.get_value_buffer(i), loc)
    out[f"ring.{i}"] = (kvp.get_qsa_key_state_buffer(i), None)
    out[f"ckv.{i}"] = (kvp.get_qsa_compressed_k_buffer(i), None)


def target_regions(kinds, forward_batch, rows: int, rtp, kvp) -> dict:
    """name -> (tensor, index) of what a verify step writes (index None: all of it).

    The KV rows are the step's ``out_cache_loc``; every other region is taken
    whole (scratch, rings and compressed keys are small next to the KV), as
    SGLang's forward writes them at rows of its own plan."""
    loc = forward_batch.out_cache_loc[:rows].long()
    out = {"rope": (kvp.qsa_rope_position_buffer, None)}
    for i, kind in enumerate(kinds):
        if kind == "qsa":
            _qsa_regions(out, i, kvp, loc)
        else:
            cache = rtp.mamba2_layer_cache(i)
            out[f"gdn_ssm.{i}"] = (cache.intermediate_ssm, None)
            out[f"gdn_conv.{i}"] = (cache.intermediate_conv_window[0], None)
    out["ple"] = (rtp.short_conv_layer_intermediate_cache(PLE_LAYER), None)
    out["ngram"] = (rtp.ngram_pool.intermediate_context, None)
    # SGLang's verify also shifts the committed conv states in place (its accept
    # step writes them again from the windows); put back, not compared
    slots = rtp.req_index_to_mamba_index_mapping[
        forward_batch.req_pool_indices.long()
    ].long()
    for i, kind in enumerate(kinds):
        if kind != "qsa":
            out[f"{RESTORE_ONLY}conv.{i}"] = (rtp.mamba2_layer_cache(i).conv[0], slots)
    return out


def draft_regions(layer_id: int, forward_batch, rows: int, kvp) -> dict:
    loc = forward_batch.out_cache_loc[:rows].long()
    out = {"rope": (kvp.qsa_rope_position_buffer, None)}
    _qsa_regions(out, layer_id, kvp, loc)
    return out


def snapshot(regions: dict) -> dict:
    return {
        n: (t if idx is None else t[idx]).clone() for n, (t, idx) in regions.items()
    }


def restore(regions: dict, saved: dict) -> None:
    for n, (t, idx) in regions.items():
        if idx is None:
            t.copy_(saved[n])
        else:
            t[idx] = saved[n]


def _metrics(got: torch.Tensor, want: torch.Tensor, pre=None) -> dict:
    """``rows``: the rows SGLang's forward wrote, the ones compared; ``extra``: the
    rows only the launches wrote. ``rel`` and ``cosine`` over all compared rows;
    ``row_cosine``: the least row's (reported, the worst case of one row)."""
    o, r = got.reshape(got.shape[0], -1), want.reshape(want.shape[0], -1)
    if pre is None:
        rows = torch.ones(o.shape[0], dtype=torch.bool, device=o.device)
        out = {}
    else:
        p = pre.reshape(pre.shape[0], -1)
        rows = (r != p).any(1)
        out = {"rows": int(rows.sum()), "extra": int(((o != p).any(1) & ~rows).sum())}
    if not got.is_floating_point():
        d = (o[rows].long() - r[rows].long()).abs()
        return out | {"max_abs": d.max().item() if d.numel() else 0}
    if not rows.any():
        return out | {"rel": 0.0, "cosine": 1.0, "row_cosine": 1.0}
    o, r = o[rows].float(), r[rows].float()
    rel = ((o - r).norm() / r.norm().clamp_min(1e-30)).item()
    cos = torch.nn.functional.cosine_similarity(o.flatten(), r.flatten(), dim=0).item()
    live = (o.norm(dim=1) > 0) | (r.norm(dim=1) > 0)
    row_cos = (
        torch.nn.functional.cosine_similarity(o[live], r[live], dim=-1).min().item()
        if live.any()
        else 1.0
    )
    return out | {"rel": rel, "cosine": cos, "row_cosine": row_cos}


def compare(step: str, regions: dict, ref: dict, pre: dict, outputs: dict) -> dict:
    """``{step}.{output}.{metric}`` -> the worst value; ``outputs``: name -> (got, want)."""
    values: dict = {}

    def worst(key, metric, v):
        k = f"{step}.{key}.{metric}"
        if metric in ("cosine", "row_cosine"):
            values[k] = min(values.get(k, v), v)
        elif metric in ("rows", "extra"):
            values[k] = values.get(k, 0) + v
        else:
            values[k] = max(values.get(k, v), v)

    for name, (got, want) in outputs.items():
        for metric, v in _metrics(got, want).items():
            if metric not in ("rows", "extra"):
                worst(name, metric, v)
    for name, (t, idx) in regions.items():
        if name.startswith(RESTORE_ONLY):
            continue
        got, want, before = t if idx is None else t[idx], ref[name], pre[name]
        if idx is None and name.split(".")[0] in DUMMY_ROW_REGIONS:
            got, want, before = got[1:], want[1:], before[1:]
        kind, _, layer = name.partition(".")
        for metric, v in _metrics(got, want, before).items():
            worst(kind, metric, v)
            if layer.split(".")[0].isdigit() and int(layer.split(".")[0]) < FIRST_BLOCK:
                worst(f"first_{kind}", metric, v)
    return values


def report(label: str, values: dict, numerics) -> None:
    logger.info(
        "monokernel check %s: %s",
        label,
        " ".join(f"{k}={v:.6g}" for k, v in values.items()),
    )
    for name, bound in numerics.bounds.items():
        v = values.get(name)
        if v is None:
            continue
        ok = v >= bound if name.endswith("cosine") else v <= bound
        if not ok:
            logger.error(
                "monokernel check violation: %s %s = %.6g, bound %g",
                label,
                name,
                v,
                bound,
            )
