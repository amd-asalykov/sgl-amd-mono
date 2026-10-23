# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8-Flash-Next weights the launches read (RFC #43949).

In place, as SGLang's loader and its gfx950 AITER weight processing leave them:
the MXFP4 experts (FusedMoE ``w13`` / ``w2``, the shared expert fused at index
512) and the Quark W8A8 FP8 projections (``shuffle_weight(w, (16, 16))``, F32
per-channel scales). Copied, and logged: the bf16 tensors the launches read in
their own MFMA tiles (hyper-connection mixers, router, PLE projections, MTP input
fusion; packed at build from the checkpoint tensors ``read_checkpoint`` keeps) and
the GDN ``in_proj_ba``, which SGLang keeps row-major (96 rows, not a multiple of
64), padded to 128 rows and shuffled.

The layout SGLang leaves is checked byte for byte against the checkpoint on a
few experts and projections (``check_*``): an AITER layout change is a refusal
at startup, not a wrong token.
"""

from __future__ import annotations

import glob
import os
import re

import torch

from sgl_amd_mono.qwen38_flash_next.gfx950 import layout as LY
from sgl_amd_mono.qwen38_flash_next.gfx950 import packing
from sgl_amd_mono.qwen38_flash_next.gfx950.config import N_EXPERTS

SGLANG_READS = {
    "srt/layers/quantization/fp8_utils.py": ("use_aiter_bpreshuffle_gemm",),
    # Qwen4ExpModel and its decoder layers (Qwen3.5 / Qwen3-Next modules)
    "srt/models/qwen3_next.py": (
        "layers",
        "mlp",
        "linear_attn",
        "in_proj_qkvz",
        "in_proj_ba",
        "out_proj",
        "qkv_proj",
        "o_proj",
        "embed_tokens",
    ),
    "srt/models/qwen2_moe.py": ("experts",),
    "srt/layers/attention/qsa/qsa_indexer.py": ("index_qk_proj",),
    "srt/models/qwen4_exp.py": (
        "indexer",
        "ple",
        "ple_embedding",
        "ngram_embedding",
        '"layer_multipliers"',
        '"ngram_heads_vocab_sizes"',
        '"ngram_heads_offsets"',
        '"weight_scale"',
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": ("num_fused_shared_experts",),
    "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py": (
        '"w13_weight"',
        '"w2_weight"',
        '"w13_weight_scale"',
        '"w2_weight_scale"',
        "is_shuffled",
    ),
}

TARGET_PFX = "model.language_model."
DRAFT_PFX = "mtp."
# the checkpoint tensors the target launches pack (per layer; the PLE on layer 1)
_HC = ("attn_hyper_connection", "mlp_hyper_connection")
_HC_W = (
    "hc_norm",
    "input_mix_weight_down",
    "input_mix_weight_up",
    "block_inject_weight",
)
_GDN_SMALL = ("A_log", "conv1d.weight", "dt_bias", "norm.weight")
_QSA_NORMS = ("q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm")
_PLE = ("key_proj", "value_proj", "norm_key", "norm_query", "norm_conv", "conv1d")
_FINAL = ("hc_norm", "input_mix_weight_down", "input_mix_weight_up")


def _layer_small_names(kind: str) -> list[str]:
    names = [f"{hc}.{w}.weight" for hc in _HC for w in _HC_W]
    if kind == "gdn":
        names += [f"linear_attn.{w}" for w in _GDN_SMALL]
    else:
        names += [f"self_attn.{w}.weight" for w in _QSA_NORMS]
    return names + ["mlp.gate.weight", "mlp.shared_expert_gate.weight"]


# ------------------------------------------------------------------ checkpoint
def checkpoint_dir(path: str) -> str:
    """The local directory of ``path`` (a directory, or a hub id already downloaded)."""
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        path, allow_patterns=["*.safetensors"], local_files_only=True
    )


def read_checkpoint(copies: dict, path: str, keep) -> None:
    """``keep(copies, name, tensor)`` for every tensor of the checkpoint under
    ``path``; ``tensor`` loads it, so only the tensors kept are read."""
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(checkpoint_dir(path), "*.safetensors")))
    if not files:
        raise ValueError(f"no safetensors files under {path}")
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as fh:
            for name in fh.keys():
                keep(copies, name, lambda fh=fh, name=name: fh.get_tensor(name))


def _cpu(tensor) -> torch.Tensor:
    """A kept tensor, on the CPU: ``tensor`` or what the loader ``tensor`` reads."""
    t = tensor() if callable(tensor) else tensor
    return t.detach().to("cpu", copy=True)


# the experts and projections compared byte for byte with SGLang's (target layers
# 0 and 3, and the MTP layer)
CHECK_LAYERS = (0, 3)
CHECK_EXPERTS = (0, 255, 511, N_EXPERTS)  # N_EXPERTS: the shared expert
_EXPERT = re.compile(
    r"mlp\.(?:experts\.(?P<e>\d+)|(?P<shared>shared_expert))\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<w>weight|weight_scale)"
)
_FP8 = re.compile(
    r"(?:linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"
    r"|self_attn\.(?:q_proj|k_proj|v_proj|indexer\.index_qk_proj|o_proj))\.(?:weight|weight_scale)"
)


def keep_checkpoint_tensor(copies: dict, layer, rest: str, tensor: torch.Tensor,
                           kinds: dict) -> None:  # fmt: skip
    """Keep ``rest`` (a layer tensor's name below its layer prefix) of ``layer``
    when the launches pack it (``("pack", layer, rest)``) or a startup check reads
    it (``("check", layer, rest)``), on the CPU until build."""
    kind = kinds.get(layer)
    if kind is None:
        return
    names = _layer_small_names(kind)
    if layer == 1 and kind == "gdn":
        names += [f"ple.{n}.weight" for n in _PLE]
    if rest in names:
        copies["pack", layer, rest] = _cpu(tensor)
        return
    if layer not in CHECK_LAYERS:
        return
    m = _EXPERT.fullmatch(rest)
    if m is not None:
        e = N_EXPERTS if m["shared"] else int(m["e"])
        if e in CHECK_EXPERTS:
            copies["check", layer, rest] = _cpu(tensor)
    elif _FP8.fullmatch(rest):
        copies["check", layer, rest] = _cpu(tensor)


# ------------------------------------------------------------------ SGLang params
def moe_params(experts) -> dict:
    """The FusedMoE MXFP4 params the kernel reads in place."""
    return {
        "gu": experts.w13_weight.data,
        "gu_s": experts.w13_weight_scale.data.view(torch.uint8),
        "dn": experts.w2_weight.data,
        "dn_s": experts.w2_weight_scale.data.view(torch.uint8),
    }


def moe_reasons(experts, where: str) -> list[str]:
    reasons = []
    if experts.num_fused_shared_experts != 1:
        reasons.append(f"{where}: the shared expert is not fused into the experts")
    for w in (experts.w13_weight, experts.w2_weight):
        if w.dtype != torch.uint8 or not getattr(w, "is_shuffled", False):
            reasons.append(f"{where}: the experts are not MXFP4 in AITER's shuffle")
            return reasons
    try:
        packing.check_moe_params(moe_params(experts))
    except AssertionError as e:
        reasons.append(f"{where}: expert tensors {e}")
    return reasons


# kernel name -> (SGLang param below the decoder layer, rows, K, checkpoint tensors it merges)
GDN_PROJ = {
    "ip": ("linear_attn.in_proj_qkvz", 16384, 2560,
           ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")),
    "out": ("linear_attn.out_proj", 2560, 6144, ("linear_attn.out_proj",)),
    "ba": ("linear_attn.in_proj_ba", LY.BA_ROWS, 2560,
           ("linear_attn.in_proj_b", "linear_attn.in_proj_a")),  # 96 real rows
}  # fmt: skip
# the full-attention decoder layer holds qkv_proj / o_proj itself and the indexer as .indexer
QSA_PROJ = {
    "qp": ("qkv_proj", 13312, 2560, ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
    "qp2": ("indexer.index_qk_proj", 640, 2560, ("self_attn.indexer.index_qk_proj",)),
    "out": ("o_proj", 2560, 6144, ("self_attn.o_proj",)),
}  # fmt: skip


def _fp8_param(layer, name: str, rows: int, k: int):
    """(weight [N, K] as stored, F32 scales [N]) of an FP8 linear, or a reason."""
    params = dict(layer.named_parameters())
    if name + ".weight" not in params or name + ".weight_scale" not in params:
        return f"{name}: no FP8 weight / weight_scale"
    w, sc = params[name + ".weight"].data, params[name + ".weight_scale"].data
    if (
        w.dtype != torch.float8_e4m3fn
        or sc.dtype != torch.float32
        or not sc.is_contiguous()
    ):
        return f"{name}: {w.dtype} weight / {sc.dtype} scale, not Quark W8A8 FP8 per channel"
    n = sc.numel()
    wt = (
        w.t() if tuple(w.shape) == (k, n) else w
    )  # SGLang keeps [N, K] behind a [K, N] view
    if tuple(wt.shape) != (n, k) or not wt.is_contiguous() or n > rows:
        return f"{name}: weight {tuple(w.shape)} / stride {w.stride()}"
    return wt, sc.reshape(-1)


def proj_params(layer, table: dict, copies: dict, layer_tag) -> tuple[dict, list[str]]:
    """{kernel name: (FP8 weight in AITER's (16, 16) shuffle, F32 scales [rows])}:
    SGLang's own when it keeps them shuffled at the kernel's rows, else a padded,
    shuffled copy (``copies["fp8 in_proj_ba", layer]``)."""
    from aiter.ops.shuffle import shuffle_weight
    from sglang.srt.layers.quantization.fp8_utils import use_aiter_bpreshuffle_gemm

    out, reasons = {}, []
    for key, (name, rows, k, _) in table.items():
        got = _fp8_param(layer, name, rows, k)
        if isinstance(got, str):
            reasons.append(got)
            continue
        wt, sc = got
        n = sc.numel()
        if n == rows and use_aiter_bpreshuffle_gemm(n):
            out[key] = (wt, sc)
            continue
        if use_aiter_bpreshuffle_gemm(n):
            reasons.append(f"{name}: {n} rows shuffled, the kernel reads {rows}")
            continue
        kw = torch.zeros(rows, k, dtype=wt.dtype, device=wt.device)
        kw[:n] = wt
        kw = shuffle_weight(kw, (16, 16))
        ks = torch.ones(rows, dtype=torch.float32, device=sc.device)
        ks[:n] = sc
        copies[
            f"fp8 {name.rsplit('.', 1)[-1]} (padded, shuffled)", layer_tag, "weight"
        ] = kw
        copies[
            f"fp8 {name.rsplit('.', 1)[-1]} (padded, shuffled)", layer_tag, "scale"
        ] = ks
        out[key] = (kw, ks)
    return out, reasons


# ------------------------------------------------------------------ byte checks
def _unshuffle_scales(s):
    """aiter's e8m0_shuffle inverse on one expert's [rows, groups]."""
    rows, g = s.shape
    return (
        s.reshape(rows // 32, g // 8, 4, 16, 2, 2)
        .permute(0, 5, 3, 1, 4, 2)
        .reshape(rows, g)
    )


def check_experts(
    moe: dict, copies: dict, layer_tag, where: str, ids=CHECK_EXPERTS
) -> list[str]:
    """SGLang's AITER-layout experts against the checkpoint bytes read_checkpoint kept."""
    reasons = []
    for e in ids:
        src = "mlp.shared_expert" if e == N_EXPERTS else f"mlp.experts.{e}"
        got = packing.unshuffle_moe_expert(moe, e)
        names = [f"{src}.{p}.{w}" for w in ("weight", "weight_scale")
                 for p in ("gate_proj", "up_proj", "down_proj")]  # fmt: skip
        if any(("check", layer_tag, n) not in copies for n in names):
            reasons.append(
                f"{where}: expert {e}'s checkpoint tensors were not seen at load"
            )
            continue
        for n, g in zip(names, got):
            ref = copies["check", layer_tag, n].view(torch.uint8)
            if not torch.equal(g.cpu(), ref):
                reasons.append(
                    f"{where}: expert {e} {n} differs from the checkpoint in AITER's layout"
                )
    return reasons


def check_proj(
    proj: dict, table: dict, copies: dict, layer_tag, where: str
) -> list[str]:
    reasons = []
    for key, (name, rows, _, srcs) in table.items():
        names = [f"{s}.{w}" for s in srcs for w in ("weight", "weight_scale")]
        if any(("check", layer_tag, n) not in copies for n in names):
            reasons.append(
                f"{where}: {name}'s checkpoint tensors were not seen at load"
            )
            continue
        w = torch.cat([copies["check", layer_tag, f"{s}.weight"] for s in srcs]).view(
            torch.uint8
        )
        s = torch.cat(
            [
                copies["check", layer_tag, f"{x}.weight_scale"].float().reshape(-1)
                for x in srcs
            ]
        )
        kw, ks = proj[key]
        n = w.shape[0]
        got = packing.unshuffle16(kw).cpu()
        if not (torch.equal(got[:n], w) and torch.equal(ks[:n].cpu(), s)):
            reasons.append(
                f"{where}: {name} differs from the checkpoint in AITER's layout"
            )
        elif n < rows and got[n:].any():
            reasons.append(f"{where}: {name}'s padding rows are not zero")
    return reasons


def drop_checks(copies: dict) -> None:
    for k in [k for k in copies if k[0] == "check"]:
        del copies[k]


# ------------------------------------------------------------------ packing
def _layer_dict(copies: dict, layer_tag, prefix: str = "") -> dict:
    """The kept checkpoint tensors of one layer on the GPU, by name below it."""
    dev = torch.cuda.current_device()
    return {
        prefix + rest: t.to(dev)
        for (kind, tag, rest), t in copies.items()
        if kind == "pack" and tag == layer_tag
    }


def _pop_pack(copies: dict, layer_tag) -> None:
    for k in [k for k in copies if k[0] == "pack" and k[1] == layer_tag]:
        del copies[k]


def _log_packed(copies: dict, kind: str, layer_tag, t) -> None:
    """Record a packed tensor the launches read as a weight copy (the startup
    log lists every copy)."""
    if isinstance(t, torch.Tensor):
        copies[kind, layer_tag, len(copies)] = t
    elif isinstance(t, dict):
        for v in t.values():
            _log_packed(copies, kind, layer_tag, v)


def pack_target_layer(layer_mod, layer_id: int, kind: str, copies: dict):
    """One decoder layer's packed dict (``packing.pack_*_layer``), SGLang's experts
    and FP8 projections in it; the kept checkpoint tensors become packed copies."""
    moe = moe_params(layer_mod.mlp.experts)
    proj, reasons = proj_params(
        layer_mod, GDN_PROJ if kind == "gdn" else QSA_PROJ, copies, layer_id
    )
    if reasons:
        raise ValueError("; ".join(reasons))
    lw = _layer_dict(copies, layer_id)
    pack = packing.pack_gdn_layer if kind == "gdn" else packing.pack_qsa_layer
    t = pack(lw, moe=moe, proj=proj)
    ple = packing.pack_ple(lw) if layer_id == 1 else None
    _pop_pack(copies, layer_id)
    params = {p.data_ptr() for p in layer_mod.parameters()}
    for name in ("hca", "hcm", "router", "conv_w", "neg_a", "dt_bias", "norm_w",
                 "qnorm", "knorm", "iqnorm", "iknorm"):  # fmt: skip
        if name in t and not _is_param(t[name], params):
            _log_packed(
                copies,
                _COPY_KIND.get(name, "bf16 layer norms / GDN params"),
                layer_id,
                t[name],
            )
    if ple is not None:
        _log_packed(copies, "bf16 PLE (packed)", layer_id, ple)
    return t, ple


_COPY_KIND = {
    "hca": "bf16 hyper-connection mixers (packed)",
    "hcm": "bf16 hyper-connection mixers (packed)",
    "router": "bf16 router (packed)",
}


def _is_param(t, params) -> bool:
    return isinstance(t, torch.Tensor) and t.data_ptr() in params


def keep_final(copies: dict, rest: str, tensor: torch.Tensor) -> None:
    if rest in [f"hyper_connection_mixer.{w}.weight" for w in _FINAL]:
        copies["pack", "final", rest] = _cpu(tensor)


def pack_final(copies: dict) -> dict:
    lw = _layer_dict(copies, "final")
    p = "hyper_connection_mixer."
    final = {
        "norm": lw[p + "hc_norm.weight"].contiguous(),
        "down": packing.pack_hc_down(
            lw[p + "input_mix_weight_down.weight"]
        ),  # no combine
        "up": packing.pack_bf16(lw[p + "input_mix_weight_up.weight"].contiguous()),
    }
    _pop_pack(copies, "final")
    _log_packed(copies, "bf16 hyper-connection mixers (packed)", "final", final)
    return final


def ngram_tables(ple_layer) -> tuple:
    """SGLang's PLE n-gram table (FP8, on the GPU), its scale and the hash meta."""
    emb = ple_layer.ple_embedding
    meta = (
        torch.cat(
            [
                emb.layer_multipliers,
                emb.ngram_heads_vocab_sizes,
                emb.ngram_heads_offsets,
            ]
        )
        .to(device=torch.cuda.current_device(), dtype=torch.int64)
        .contiguous()
    )
    return emb.ngram_embedding.weight, emb.ngram_embedding.weight_scale, meta


def ngram_reasons(ple_layer) -> list[str]:
    emb = ple_layer.ple_embedding
    table = emb.ngram_embedding.weight
    reasons = []
    if (
        table.dtype != torch.float8_e4m3fn
        or table.shape[1] != LY.NGRAM_DIM
        or not table.is_cuda
    ):
        reasons.append(
            f"the PLE n-gram table is {table.dtype} {tuple(table.shape)} on {table.device}; "
            f"the kernels read FP8 [rows, {LY.NGRAM_DIM}] on the GPU (--no-ple-offload-embedding)"
        )
    n = (
        emb.layer_multipliers.numel()
        + emb.ngram_heads_vocab_sizes.numel()
        + emb.ngram_heads_offsets.numel()
    )
    if n != 3 + 2 * LY.NGRAM_HEADS:
        reasons.append(
            f"the PLE n-gram hash has {n} meta values, not {3 + 2 * LY.NGRAM_HEADS}"
        )
    return reasons


def pack_draft(mtp_model, copies: dict) -> dict:
    """The MTP head packed (``packing.pack_mtp``): SGLang's experts and FP8
    projections, the kept checkpoint tensors as packed copies."""
    layer = mtp_model.model.layers[0]
    moe = moe_params(layer.mlp.experts)
    proj, reasons = proj_params(layer, QSA_PROJ, copies, "mtp")
    if reasons:
        raise ValueError("; ".join(reasons))
    lw = _layer_dict(copies, "mtp")
    packed = packing.pack_mtp(lw, moe=moe, proj=proj)
    _pop_pack(copies, "mtp")
    m = packed["layer"]
    for name in ("hca", "hcm", "router"):
        _log_packed(copies, _COPY_KIND[name], "mtp", m[name])
    _log_packed(copies, "bf16 hyper-connection mixers (packed)", "mtp", packed["final"])
    _log_packed(copies, "bf16 MTP input fusion (packed)", "mtp",
                {k: packed[k] for k in ("pe", "ph", "fce", "fch")})  # fmt: skip
    _log_packed(copies, "bf16 layer norms / GDN params", "mtp",
                {k: m[k] for k in ("qnorm", "knorm", "iqnorm", "iknorm")})  # fmt: skip
    return packed


DRAFT_CHECK_EXPERTS = (0, 511, N_EXPERTS)
# the MTP head's checkpoint tensors the draft launch packs (below "mtp.")
DRAFT_TOP = (
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "fc_embedding.weight",
    "fc_hidden.weight",
) + tuple(f"hyper_connection_mixer.{w}.weight" for w in _FINAL)


def keep_draft_tensor(copies: dict, rest: str, tensor: torch.Tensor) -> None:
    """``rest``: a checkpoint name below ``mtp.``."""
    if rest in DRAFT_TOP:
        copies["pack", "mtp", rest] = _cpu(tensor)
        return
    if not rest.startswith("layers.0."):
        return
    sub = rest[len("layers.0.") :]
    if sub in _layer_small_names("qsa"):
        copies["pack", "mtp", rest] = _cpu(tensor)
        return
    m = _EXPERT.fullmatch(sub)
    if m is not None:
        e = N_EXPERTS if m["shared"] else int(m["e"])
        if e in DRAFT_CHECK_EXPERTS:
            copies["check", "mtp", sub] = _cpu(tensor)
    elif _FP8.fullmatch(sub):
        copies["check", "mtp", sub] = _cpu(tensor)


def draft_unseen(copies: dict) -> list[str]:
    want = list(DRAFT_TOP) + [f"layers.0.{n}" for n in _layer_small_names("qsa")]
    return [n for n in want if ("pack", "mtp", n) not in copies]


def target_unseen(copies: dict, kinds) -> list[str]:
    missing = []
    for i, kind in enumerate(kinds):
        names = _layer_small_names(kind) + (
            [f"ple.{n}.weight" for n in _PLE] if i == 1 else []
        )
        missing += [f"layers.{i}.{n}" for n in names if ("pack", i, n) not in copies]
    missing += [f"hyper_connection_mixer.{w}.weight" for w in _FINAL
                if ("pack", "final", f"hyper_connection_mixer.{w}.weight") not in copies]  # fmt: skip
    return missing
