# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the Qwen3.8-Flash-Next launch checks (RFC #43949): a truncated
model's real layer weights in SGLang's post-load layouts, pools of the shapes
SGLang allocates, filled with random state, and a step's batch.

``QWEN38_FLASH_NEXT_MODEL_PATH``: the checkpoint, a directory or a Hugging Face id
(default amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8; from the hub, only the
shards the fixtures read are downloaded).

The PLE n-gram table and the token embedding are random (the real n-gram table
is 51 GB); every other weight is the checkpoint's, converted as SGLang's loader
and its AITER weight processing leave it (``sglang_params``)."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from types import SimpleNamespace

import torch

from sgl_amd_mono.qwen38_flash_next.gfx950 import layout as LY
from sgl_amd_mono.qwen38_flash_next.gfx950 import packing
from sgl_amd_mono.qwen38_flash_next.gfx950.config import HC_DIM, HIDDEN, N_EXPERTS
from sgl_amd_mono.qwen38_flash_next.gfx950.runner import (
    DraftWeights,
    StepBatch,
    TargetWeights,
)

MODEL = os.environ.get(
    "QWEN38_FLASH_NEXT_MODEL_PATH", "amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8"
)
PFX = "model.language_model."
VOCAB = 4096  # the random embedding's rows: input ids are drawn below it
# request contexts (tokens before the verify rows): short, a block edge, long
# (two-level top-512 selection past 32k blocks) and two requests at one position
CONTEXTS = (60, 1203, 9000, 70001, 2048, 133, 140003, 60)
DEV = "cuda"


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEV).manual_seed(seed)


# ------------------------------------------------------------------ checkpoint
def _file(name: str) -> str:
    """A checkpoint file's local path: in MODEL's directory, or from the hub cache."""
    if os.path.isdir(MODEL):
        return os.path.join(MODEL, name)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(MODEL, name)


@lru_cache(maxsize=1)
def _index() -> dict:
    with open(_file("model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def load(names) -> dict:
    from safetensors import safe_open

    by_file: dict = {}
    for n in names:
        by_file.setdefault(_index()[n], []).append(n)
    out = {}
    for fname, ns in by_file.items():
        with safe_open(_file(fname), "pt", device="cpu") as f:
            for n in ns:
                out[n] = f.get_tensor(n).to(DEV)
    return out


def _small_names(p: str, kind: str) -> list[str]:
    names = [
        f"{p}{hc}.{w}"
        for hc in ("attn_hyper_connection", "mlp_hyper_connection")
        for w in (
            "hc_norm.weight",
            "input_mix_weight_down.weight",
            "input_mix_weight_up.weight",
            "block_inject_weight.weight",
        )
    ]
    if kind == "gdn":
        names += [
            f"{p}linear_attn.{w}"
            for w in ("A_log", "conv1d.weight", "dt_bias", "norm.weight")
        ]
    else:
        names += [
            f"{p}self_attn.{w}"
            for w in (
                "q_norm.weight",
                "k_norm.weight",
                "indexer.q_layernorm.weight",
                "indexer.k_layernorm.weight",
            )
        ]
    return names + [f"{p}mlp.gate.weight", f"{p}mlp.shared_expert_gate.weight"]


# ------------------------------------------------------------------ SGLang layouts
def sglang_moe(p: str) -> dict:
    """The FusedMoE MXFP4 params as SGLang keeps them on gfx950 (AITER): routed
    experts and the shared one (index 512) stacked, the intermediate dim padded
    640 -> 768, ``shuffle_weight(w, (16, 16))`` and ``e8m0_shuffle``'d scales."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    I, IP = LY.MOE_INTER, LY.MOE_INTER_PAD
    srcs = [f"{p}mlp.experts.{e}." for e in range(N_EXPERTS)] + [
        f"{p}mlp.shared_expert."
    ]
    raw = load(
        [
            s + f"{proj}.{w}"
            for s in srcs
            for proj in ("gate_proj", "up_proj", "down_proj")
            for w in ("weight", "weight_scale")
        ]
    )

    def stack(proj, w):
        return torch.stack([raw[s + f"{proj}.{w}"].view(torch.uint8) for s in srcs])

    def pad_gu(g, u):
        out = torch.zeros(g.shape[0], 2 * IP, g.shape[2], dtype=torch.uint8, device=DEV)
        out[:, :I], out[:, IP : IP + I] = g, u
        return out

    def pad_k(t, n):
        out = torch.zeros(*t.shape[:-1], n, dtype=torch.uint8, device=DEV)
        out[..., : t.shape[-1]] = t
        return out

    def sh_s(sc):
        e, r, _ = sc.shape
        return e8m0_shuffle(sc.reshape(e * r, -1).contiguous()).view(e, r, -1)

    w13 = pad_gu(stack("gate_proj", "weight"), stack("up_proj", "weight"))
    s13 = pad_gu(stack("gate_proj", "weight_scale"), stack("up_proj", "weight_scale"))
    w2 = pad_k(stack("down_proj", "weight"), IP // 2)
    s2 = pad_k(stack("down_proj", "weight_scale"), IP // 32)
    raw = None  # the stacked copies are kept; free the rest
    return {
        "gu": shuffle_weight(w13.contiguous(), (16, 16)),
        "gu_s": sh_s(s13),
        "dn": shuffle_weight(w2.contiguous(), (16, 16)),
        "dn_s": sh_s(s2),
    }


def sglang_fp8(p: str, names, rows: int):
    """A merged Quark W8A8 FP8 projection as SGLang keeps it on gfx950: rows
    ``rows`` (zero-padded, scale 1) in ``shuffle_weight(w, (16, 16))`` and F32
    per-channel scales."""
    from aiter.ops.shuffle import shuffle_weight

    raw = load([f"{p}{n}.{w}" for n in names for w in ("weight", "weight_scale")])
    w = torch.cat([raw[f"{p}{n}.weight"] for n in names])
    s = torch.cat([raw[f"{p}{n}.weight_scale"].float().reshape(-1) for n in names])
    if w.shape[0] < rows:
        pad = rows - w.shape[0]
        w = torch.cat([w, torch.zeros(pad, w.shape[1], dtype=w.dtype, device=DEV)])
        s = torch.cat([s, torch.ones(pad, dtype=s.dtype, device=DEV)])
    return shuffle_weight(w.contiguous(), (16, 16)), s.contiguous()


def sglang_proj(p: str, kind: str) -> dict:
    if kind == "gdn":
        a = "linear_attn."
        return {
            "ip": sglang_fp8(p, (a + "in_proj_qkv", a + "in_proj_z"), 16384),
            "out": sglang_fp8(p, (a + "out_proj",), 2560),
            "ba": sglang_fp8(p, (a + "in_proj_b", a + "in_proj_a"), LY.BA_ROWS),
        }
    a = "self_attn."
    return {
        "qp": sglang_fp8(p, (a + "q_proj", a + "k_proj", a + "v_proj"), 13312),
        "qp2": sglang_fp8(p, (a + "indexer.index_qk_proj",), 640),
        "out": sglang_fp8(p, (a + "o_proj",), 2560),
    }


def kinds_of(n_layers: int) -> tuple[str, ...]:
    return tuple("qsa" if i % 4 == 3 else "gdn" for i in range(n_layers))


def _ngram_tables():
    """A small random n-gram table with the checkpoint's hash multipliers."""
    g = _gen(7)
    vocab = torch.tensor(
        [1009 + 2 * h for h in range(LY.NGRAM_HEADS)], dtype=torch.int64
    )
    offsets = torch.cumsum(vocab, 0) - vocab
    mult = torch.tensor(
        [23703573157769, 20109073645365, 8052911324071], dtype=torch.int64
    )
    rows = int(vocab.sum())
    table = (torch.randn(rows, LY.NGRAM_DIM, generator=g, device=DEV) * 2).to(
        torch.float8_e4m3fn
    )
    scale = torch.tensor([2.0e-4], dtype=torch.bfloat16, device=DEV)
    meta = torch.cat([mult, vocab, offsets]).to(device=DEV, dtype=torch.int64)
    return table, scale, meta


def target_weights(n_layers: int = 12) -> TargetWeights:
    """Layers 0 .. n_layers - 1 of the checkpoint (a multiple of 4), the final
    mixer, a random embedding and n-gram table."""
    kinds = kinds_of(n_layers)
    layers = {}
    ple = None
    for i, kind in enumerate(kinds):
        p = f"{PFX}layers.{i}."
        names = _small_names(p, kind)
        if i == 1:
            names += [
                f"{p}ple.{n}"
                for n in (
                    "key_proj.weight",
                    "value_proj.weight",
                    "norm_key.weight",
                    "norm_query.weight",
                    "norm_conv.weight",
                    "conv1d.weight",
                )
            ]
        raw = load(names)
        lw = {k[len(p) :]: v for k, v in raw.items()}
        pack = packing.pack_gdn_layer if kind == "gdn" else packing.pack_qsa_layer
        layers[i] = pack(lw, moe=sglang_moe(p), proj=sglang_proj(p, kind))
        if i == 1:
            ple = packing.pack_ple(lw)
        torch.cuda.empty_cache()
    mix = load(
        [
            f"{PFX}hyper_connection_mixer.{n}.weight"
            for n in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up")
        ]
    )
    final = {
        "norm": mix[f"{PFX}hyper_connection_mixer.hc_norm.weight"].contiguous(),
        "down": packing.pack_hc_down(
            mix[f"{PFX}hyper_connection_mixer.input_mix_weight_down.weight"]
        ),
        "up": packing.pack_bf16(
            mix[f"{PFX}hyper_connection_mixer.input_mix_weight_up.weight"].contiguous()
        ),
    }
    table, scale, meta = _ngram_tables()
    embed = (
        torch.randn(VOCAB, HIDDEN, generator=_gen(3), device=DEV) * 0.02
    ).bfloat16()
    return TargetWeights(
        kinds=kinds, layers=layers, ple=ple, final=final, embed_w=embed,
        ngram_table=table, ngram_scale=scale, ngram_meta=meta, eos_token_id=VOCAB - 1,
    )  # fmt: skip


def draft_weights() -> DraftWeights:
    """The checkpoint's MTP head (``mtp.*``) and a random embedding."""
    idx = _index()
    names = [k for k in idx if k.startswith("mtp.") and ".mlp.experts." not in k
             and ".mlp.shared_expert." not in k
             and not any(f".self_attn.{q}.weight" in k for q in
                         ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj", "o_proj"))]  # fmt: skip
    lw = {k[len("mtp.") :]: v for k, v in load(names).items()}
    p = "mtp.layers.0."
    packed = packing.pack_mtp(lw, moe=sglang_moe(p), proj=sglang_proj(p, "qsa"))
    embed = (
        torch.randn(VOCAB, HIDDEN, generator=_gen(5), device=DEV) * 0.02
    ).bfloat16()
    return DraftWeights(
        packed=packed, layer_id=0, embed_w=embed, eos_token_id=VOCAB - 1
    )


# ------------------------------------------------------------------ pools
class Pools:
    """SGLang's pools as the runners read them (req_to_token pool and KV pool
    accessors), for ``R_max`` requests of ``L`` verify rows over ``contexts``.

    Request r sits at request slot 2 r + 3 and mamba slot 5 R_max - 2 r; its
    position p at KV slot base_r + p."""

    def __init__(self, kinds, L: int, contexts=CONTEXTS, seed: int = 0, ssm_bf16=True):
        g = _gen(seed)
        R = len(contexts)
        self.L, self.contexts, self.kinds = L, tuple(contexts), tuple(kinds)
        self.reqs = [2 * r + 3 for r in range(R)]
        self.slots = [5 * R - 2 * r for r in range(R)]
        n_req = max(self.reqs) + 2
        bases, off = [], 64
        for c in contexts:
            bases.append(off)
            off += (c + L + 64 + 63) // 64 * 64
        self.bases, ntok = bases, off + 64
        stride = (max(contexts) + L + 64 + 255) // 256 * 256
        self.req_to_token = torch.zeros(n_req, stride, dtype=torch.int32, device=DEV)
        for r, c in enumerate(contexts):
            self.req_to_token[self.reqs[r], : c + L] = bases[r] + torch.arange(
                c + L, dtype=torch.int32, device=DEV
            )
        self.req_index_to_mamba_index_mapping = torch.zeros(
            n_req, dtype=torch.int32, device=DEV
        )
        for r in range(R):
            self.req_index_to_mamba_index_mapping[self.reqs[r]] = self.slots[r]
        n_slot = 5 * R + 2
        ssm_dt = torch.bfloat16 if ssm_bf16 else torch.float32

        def rnd(*shape, scale=1.0, dtype=torch.bfloat16):
            return (torch.randn(*shape, generator=g, device=DEV) * scale).to(dtype)

        self.mamba = {}
        self.kv = {}
        for i, kind in enumerate(kinds):
            if kind == "gdn":
                self.mamba[i] = SimpleNamespace(
                    conv=[rnd(n_slot, HC_DIM, 3, scale=0.5)],
                    temporal=rnd(n_slot, 48, 128, 128, scale=0.02, dtype=ssm_dt),
                    intermediate_ssm=torch.zeros(
                        R, L, 48, 128, 128, dtype=ssm_dt, device=DEV
                    ),
                    intermediate_conv_window=[
                        torch.zeros(R, L, HC_DIM, 3, dtype=torch.bfloat16, device=DEV)
                    ],
                )
            else:
                k = torch.zeros(ntok, 2, 256, dtype=torch.bfloat16, device=DEV)
                v = torch.zeros_like(k)
                comp = torch.zeros(
                    ntok // 4 + 16, 1, 128, dtype=torch.bfloat16, device=DEV
                )
                ring = rnd(n_req * 4, 1, 128, scale=0.5)
                for r, c in enumerate(contexts):
                    k[bases[r] : bases[r] + c] = rnd(c, 2, 256, scale=0.5)
                    v[bases[r] : bases[r] + c] = rnd(c, 2, 256, scale=0.5)
                    comp[bases[r] // 4 : (bases[r] + c) // 4] = rnd(
                        (bases[r] + c) // 4 - bases[r] // 4, 1, 128, scale=0.5
                    )
                self.kv[i] = SimpleNamespace(k=k, v=v, ring=ring, comp=comp)
        self.qsa_rope_position_buffer = torch.zeros(
            n_req * 4, 3, dtype=torch.int64, device=DEV
        )
        for r, c in enumerate(contexts):
            for q in range(4):  # the pending keys of positions c - 4 .. c - 1
                p = c - 1 - ((c - 1 - q) % 4)
                self.qsa_rope_position_buffer[self.reqs[r] * 4 + q] = max(p, 0)
        self.short_conv = rnd(n_slot, HC_DIM, LY.PLE_STATE_LEN, scale=0.5)
        self.short_conv_ist = torch.zeros(
            R, L, HC_DIM * LY.PLE_STATE_LEN, dtype=torch.bfloat16, device=DEV
        )
        # the n-gram history lives with the mamba state: one row per mamba slot
        ctx = torch.randint(0, VOCAB - 1, (n_slot, 2), generator=g, device=DEV)
        self.ngram_pool = SimpleNamespace(
            context=ctx.to(torch.int64),
            intermediate_context=torch.zeros(R, L, 2, dtype=torch.int64, device=DEV),
        )

    # ReqToTokenPool accessors
    def mamba2_layer_cache(self, layer_id):
        return self.mamba[layer_id]

    def short_conv_layer_cache(self, layer_id):
        return self.short_conv

    def short_conv_layer_intermediate_cache(self, layer_id):
        return self.short_conv_ist

    # KV pool accessors
    def get_key_buffer(self, layer_id):
        return self.kv[layer_id].k

    def get_value_buffer(self, layer_id):
        return self.kv[layer_id].v

    def get_qsa_key_state_buffer(self, layer_id):
        return self.kv[layer_id].ring

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.kv[layer_id].comp

    def tensors(self) -> dict:
        """Every pool tensor by name, the launches' outputs among them."""
        out = {
            "rope_pos": self.qsa_rope_position_buffer,
            "short_conv": self.short_conv,
            "short_conv_ist": self.short_conv_ist,
            "ngram_ctx": self.ngram_pool.context,
            "ngram_ictx": self.ngram_pool.intermediate_context,
        }
        for i, m in self.mamba.items():
            out |= {f"L{i}.conv": m.conv[0], f"L{i}.ssm": m.temporal,
                    f"L{i}.issm": m.intermediate_ssm, f"L{i}.iconv": m.intermediate_conv_window[0]}  # fmt: skip
        for i, kv in self.kv.items():
            out |= {
                f"L{i}.k": kv.k,
                f"L{i}.v": kv.v,
                f"L{i}.ring": kv.ring,
                f"L{i}.comp": kv.comp,
            }
        return out

    def snapshot(self) -> dict:
        return {k: t.clone() for k, t in self.tensors().items()}

    def restore(self, snap: dict) -> None:
        for k, t in self.tensors().items():
            t.copy_(snap[k])


def cos_sin(n: int, theta: float = 10_000_000.0, rot: int = 64) -> torch.Tensor:
    """The bf16 RoPE table [positions, rot] (cos | sin) the kernels read."""
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, device=DEV).float() / rot))
    f = torch.outer(torch.arange(n, device=DEV, dtype=torch.float), inv)
    return torch.cat([f.cos(), f.sin()], -1).bfloat16().contiguous()


def step_batch(
    pools: Pools, R: int, L: int, seed: int = 1, mrope: bool = False
) -> StepBatch:
    """A verify (or draft) step of the first R requests, L rows each."""
    g = _gen(seed)
    ctx = pools.contexts[:R]
    pos = torch.tensor(
        [c + j for c in ctx for j in range(L)], dtype=torch.int64, device=DEV
    )
    locs = torch.tensor(
        [pools.bases[r] + c + j for r, c in enumerate(ctx) for j in range(L)],
        dtype=torch.int64, device=DEV,
    )  # fmt: skip
    return StepBatch(
        batch_size=R,
        input_ids=torch.randint(0, VOCAB - 1, (R * L,), generator=g, device=DEV),
        positions=pos.expand(3, -1).contiguous() if mrope else pos,
        out_cache_loc=locs,
        req_pool_indices=torch.tensor(pools.reqs[:R], dtype=torch.int64, device=DEV),
        seq_lens=torch.tensor(ctx, dtype=torch.int64, device=DEV),
    )


def diff(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, float]:
    """(bitwise equal, max |a - b| over finite values)."""
    same = (
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        if a.dtype != torch.int64
        else torch.equal(a, b)
    )
    if same:
        return True, 0.0
    if a.is_floating_point():
        d = (a.float() - b.float()).abs()
        return False, float(d[torch.isfinite(d)].max()) if torch.isfinite(
            d
        ).any() else float("nan")
    return False, float((a != b).sum())


def pad_batch(b: StepBatch, L: int, pads: int = 1) -> StepBatch:
    """``b`` with ``pads`` pad requests appended as SGLang pads a graph's batch:
    request slot 0 (mamba slot 0), every row at KV slot 0, a sequence of 1."""
    z = torch.zeros(pads * L, dtype=torch.int64, device=DEV)
    pos = b.positions
    pad_pos = torch.zeros(*pos.shape[:-1], pads * L, dtype=pos.dtype, device=DEV)
    return StepBatch(
        batch_size=b.batch_size + pads,
        input_ids=torch.cat([b.input_ids, z]),
        positions=torch.cat([pos, pad_pos], -1),
        out_cache_loc=torch.cat([b.out_cache_loc, z]),
        req_pool_indices=torch.cat([b.req_pool_indices, z[:pads]]),
        seq_lens=torch.cat(
            [b.seq_lens, torch.ones(pads, dtype=torch.int64, device=DEV)]
        ),
    )


def allowed_rows(pools: Pools, b: StepBatch) -> dict:
    """Pool tensor name -> the rows a step of ``b`` may write (None: none): its KV
    slots and their compressed groups, its requests' ring and RoPE slots, and its
    scratch rows; the committed states are only read."""
    locs = b.out_cache_loc.long().unique()
    ring = (
        b.req_pool_indices.long()[:, None] * 4 + torch.arange(4, device=DEV)
    ).flatten()
    scratch = torch.arange(b.batch_size, device=DEV)
    out = {"rope_pos": ring, "short_conv": None, "short_conv_ist": scratch,
           "ngram_ctx": None, "ngram_ictx": scratch}  # fmt: skip
    for i in pools.mamba:
        out |= {f"L{i}.conv": None, f"L{i}.ssm": None, f"L{i}.issm": scratch, f"L{i}.iconv": scratch}  # fmt: skip
    for i in pools.kv:
        out |= {f"L{i}.k": locs, f"L{i}.v": locs, f"L{i}.ring": ring, f"L{i}.comp": (locs // 4).unique()}  # fmt: skip
    return out


def stray_rows(got: dict, snap: dict, allowed: dict) -> list[str]:
    """``name rows`` of every pool tensor row the step changed outside ``allowed``."""
    bad = []
    for name, rows in allowed.items():
        changed = (got[name] != snap[name]).reshape(snap[name].shape[0], -1).any(1)
        if rows is not None:
            changed[rows] = False
        if changed.any():
            bad.append(f"{name} {changed.nonzero().flatten()[:4].tolist()}")
    return bad


def pad_rows(pools: Pools, b: StepBatch, real: int) -> dict:
    """Pool tensor name -> the rows of ``b``'s pad requests (past the first ``real``):
    the dummy slots they all write at once (unspecified, as in SGLang) and their own
    scratch rows, which read them."""
    pad = StepBatch(
        batch_size=b.batch_size - real,
        input_ids=b.input_ids[real * pools.L :],
        positions=b.positions[..., real * pools.L :],
        out_cache_loc=b.out_cache_loc[real * pools.L :],
        req_pool_indices=b.req_pool_indices[real:],
        seq_lens=b.seq_lens[real:],
    )
    rows = allowed_rows(pools, pad)
    scratch = torch.arange(real, b.batch_size, device=DEV)
    for name in rows:
        if name in ("short_conv_ist", "ngram_ictx") or name.endswith(
            (".issm", ".iconv")
        ):
            rows[name] = scratch
    return rows
