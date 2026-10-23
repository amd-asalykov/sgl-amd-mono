# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next's MTP draft on gfx950: its spec and forward (RFC #43949).

A NEXTN draft step of R <= 8 requests runs as one launch (``gfx950/runner.py``):
the input fusion of the target's residual streams and the token embedding, the
draft's full-attention layer (FP8 projections, MXFP4 routed and shared experts)
and the draft mixer; SGLang's logits processor closes the step. Draft extend
runs R x num_draft_tokens rows (``DRAFT_EXTEND_V2``), each draft decode step R
rows (``DECODE`` on the draft's runner).

SGLang's monokernel runtime routes DECODE steps of a non-speculative server and
TARGET_VERIFY steps (``runtime.shape_key``); a NEXTN draft runs neither as a
routed key: its runtime key is a verify key the draft never takes. Until SGLang
routes a draft's own steps, ``_route_draft_steps`` puts the draft-step switch in
front of the model's forward at build, before the runtime installs its own over
it: a step is routed by its forward mode, batch size and rows alone, so a
captured graph replays the decision made at capture, as a routed step does.
"""

from __future__ import annotations

import functools
import logging
import os

import torch

try:
    from sglang.kernels.plugin_kernels.amd.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.qwen38_flash_next import spec as target
from sgl_amd_mono.qwen38_flash_next import text_config_reasons

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        "get_server_args",
        "get_spec",
        '"tp_size"',
        '"pp_size"',
    ),
    "srt/arg_groups/fields/*.py": (
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
        "speculative_draft_model_path",
        "model_path",
    ),
    "srt/model_executor/forward_context.py": (
        "get_req_to_token_pool",
        "get_token_to_kv_pool",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "is_decode",
        "is_draft_extend_v2",
        "spec_info",
        "batch_size",
    ),
    "srt/mem_cache/memory_pool.py": ("req_to_token",),
    "srt/layers/rotary_embedding/base.py": ("cos_sin_cache_as",),
    # Qwen4ExpForCausalLMMTP.forward, which the draft step stands in for
    "srt/models/qwen4_exp_mtp.py": (
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
        "logits_processor",
        "lm_head",
        "_set_hc_logits_hidden_states",
        "model",
    ),
    "srt/models/qwen3_5.py": ("layers", "rotary_emb", "attn", "embed_tokens"),
    "srt/layers/radix_attention.py": ("layer_id",),
}

HC_DIM = 4 * 2560
# The launch against SGLang's draft forward on the same inputs (SGLANG_AMD_MONO_CHECK,
# eager draft extend and decode steps of 1 to 8 requests at k=3 and k=1; worst values
# with a margin; cosine and rel over the rows of the step): one layer, so the bounds
# are tight. The launch also writes the ring and RoPE slot of a draft decode's token,
# which SGLang's draft decode leaves alone (reported as extra rows).
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "draft.rope.max_abs": 0,
        "draft.kv.cosine": 0.999,
        "draft.ring.cosine": 0.999,
        "draft.logits.cosine": 0.85,
        "draft.logits.rel": 0.6,
        "draft.hidden.cosine": 0.85,
        "draft.hidden.rel": 0.65,
    },
)


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel, get_server_args, get_spec

    from sgl_amd_mono.qwen38_flash_next import weights as W

    reasons = text_config_reasons(hf_config, draft=True)
    p, s = get_parallel(), get_spec()
    if (p.tp_size, p.pp_size) != (1, 1):
        reasons.append("the launch runs the draft on one GPU: TP and PP 1")
    if s.speculative_eagle_topk not in (None, 1):
        reasons.append("the launch drafts one chain: --speculative-eagle-topk 1")
    if not 1 <= int(s.speculative_num_draft_tokens or 0) <= target.MAX_TOKENS:
        reasons.append(
            f"draft extend of {s.speculative_num_draft_tokens} tokens a request"
        )
    if len(model.model.layers) != 1 or model.hc_count * model.hidden_size != HC_DIM:
        reasons.append("needs one MTP layer on 4 hyper-connection streams")
        return reasons
    args = get_server_args()
    # the copies come from the checkpoint files, not a load hook: every load path keeps them
    _read_copies(copies, model, args.speculative_draft_model_path or args.model_path)
    unseen = W.draft_unseen(copies)
    if unseen:
        reasons.append(
            f"{len(unseen)} checkpoint tensors the launch packs are not in the checkpoint "
            f"(first: mtp.{unseen[0]})"
        )
    experts = model.model.layers[0].mlp.experts
    reasons += W.moe_reasons(experts, "MTP layer")
    if reasons:
        return reasons
    reasons += W.check_experts(W.moe_params(experts), copies, "mtp", "MTP layer",
                               ids=W.DRAFT_CHECK_EXPERTS)  # fmt: skip
    proj, why = W.proj_params(model.model.layers[0], W.QSA_PROJ, {}, "mtp")
    return reasons + (why or W.check_proj(proj, W.QSA_PROJ, copies, "mtp", "MTP layer"))


def _read_copies(copies: dict, model, path: str) -> None:
    """Keep what ``_keep_tensor`` keeps, from the checkpoint files under ``path``."""
    from sgl_amd_mono.qwen38_flash_next import weights as W

    W.read_checkpoint(copies, path, lambda c, name, t: _keep_tensor(c, model, name, t))


def _keep_tensor(copies: dict, model, name: str, tensor) -> None:
    from sgl_amd_mono.qwen38_flash_next import weights as W

    if name.startswith(W.DRAFT_PFX):
        W.keep_draft_tensor(copies, name[len(W.DRAFT_PFX) :], tensor)


class _Draft:
    """The draft's monokernel: the runner and the draft step."""

    def __init__(self, model, runner):
        self.model, self.runner = model, runner
        layer = model.model.layers[0]
        self.rope = layer.rotary_emb
        self.seen: set = set()
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1"
        self.standard = None  # the forward the draft-step switch wraps

    def forward(self, plan, input_ids, positions, forward_batch, *args, **kwargs):
        """A routed key of the draft's runtime (a verify key, which a NEXTN draft
        never runs): SGLang's forward."""
        m = self.model
        return type(m).forward(m, input_ids, positions, forward_batch, *args, **kwargs)

    def rows_of(self, forward_batch, input_ids, input_embeds) -> int:
        """The rows of a draft step the launch takes, or 0: a draft extend of R x L
        or a draft decode of R x 1 rows, R <= 8, on the target's bf16 streams."""
        fb, r = forward_batch, self.runner
        R = int(fb.batch_size)
        if input_embeds is not None or not 1 <= R <= r.max_reqs or fb.spec_info is None:
            return 0
        hs = fb.spec_info.hidden_states
        n = input_ids.numel()
        if (
            hs is None
            or hs.dtype != torch.bfloat16
            or hs.shape[-1] != HC_DIM
            or not hs.is_contiguous()
        ):
            return 0
        if fb.forward_mode.is_draft_extend_v2():
            ok = n == R * r.L_ext
        else:
            ok = fb.forward_mode.is_decode() and n == R
        return n if ok and hs.shape[0] == n else 0

    def step(self, input_ids, positions, forward_batch):
        """``Qwen4ExpForCausalLMMTP.forward`` for a draft step the launch takes."""
        from sglang.srt.model_executor.forward_context import (
            get_req_to_token_pool,
            get_token_to_kv_pool,
        )

        from sgl_amd_mono.qwen38_flash_next.gfx950.runner import StepBatch

        m, fb = self.model, forward_batch
        key = (fb.forward_mode.name, int(fb.batch_size), input_ids.numel())
        if key not in self.seen:
            self.seen.add(key)
            logger.info("monokernel qwen38_flash_next MTP draft runs %s", key)
        kvp, n = get_token_to_kv_pool(), input_ids.numel()
        check = None
        if self.check and not torch.cuda.is_current_stream_capturing():
            check = self._reference(input_ids, positions, fb, kvp, n)
        cos = self.rope.cos_sin_cache_as(torch.bfloat16)
        x, hc = self.runner.forward(
            StepBatch.of(fb),
            fb.spec_info.hidden_states,
            kvp,
            get_req_to_token_pool().req_to_token,
            cos,
        )
        out = m.logits_processor(input_ids, x, m.lm_head, fb)
        m._set_hc_logits_hidden_states(out, hc, fb)
        if check is not None:
            self._report(key, out, n, *check)
        return out

    def _reference(self, input_ids, positions, fb, kvp, n):
        """Check mode: SGLang's draft forward on the step's inputs; the regions it
        wrote are kept and put back as they were, for the launch."""
        from sgl_amd_mono.qwen38_flash_next import check as C

        regions = C.draft_regions(self.runner.w.layer_id, fb, n, kvp)
        pre = C.snapshot(regions)
        out = self.standard(input_ids, positions, fb)
        ref = C.snapshot(regions)
        C.restore(regions, pre)
        logits = out.next_token_logits[:n].float().clone()
        return regions, pre, ref, logits, out.hidden_states[:n].float().clone()

    def _report(self, key, out, n, regions, pre, ref, logits, hidden) -> None:
        from sgl_amd_mono.qwen38_flash_next import check as C

        values = C.compare("draft", regions, ref, pre, {
            "logits": (out.next_token_logits[:n], logits),
            "hidden": (out.hidden_states[:n], hidden),
        })  # fmt: skip
        agree = (
            (out.next_token_logits[:n].argmax(-1) == logits.argmax(-1))
            .float()
            .mean()
            .item()
        )
        C.report(f"draft {key[0]} {key[1]} x {key[2] // key[1]} (argmax agrees on {agree:.3f})",
                 values, NUMERICS)  # fmt: skip


def _route_draft_steps(model, draft: _Draft) -> None:
    """Put the draft-step switch in front of the model's forward (see the module
    docstring); every other step runs the forward it wraps."""
    standard = draft.standard = model.forward

    @functools.wraps(standard)
    def forward(input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        if draft.rows_of(forward_batch, input_ids, input_embeds):
            return draft.step(input_ids, positions, forward_batch)
        return standard(input_ids, positions, forward_batch, input_embeds, **kwargs)

    model.forward = forward


def _build(model, hf_config, ctx) -> _Draft:
    from sglang.srt.runtime_context import get_spec

    from sgl_amd_mono.qwen38_flash_next import weights as W
    from sgl_amd_mono.qwen38_flash_next.gfx950.runner import DraftRunner, DraftWeights

    copies = ctx.copies
    W.drop_checks(copies)
    layer = model.model.layers[0]
    weights = DraftWeights(
        packed=W.pack_draft(model, copies),
        layer_id=int(layer.attn.layer_id),
        embed_w=model.model.embed_tokens.weight,
        eos_token_id=int(model.config.eos_token_id),
    )
    L = int(get_spec().speculative_num_draft_tokens)
    runner = DraftRunner(weights, L, layer.mlp.experts.w13_weight.device)
    runner.set_deadline(ctx.deadline_s)
    draft = _Draft(model, runner)
    _route_draft_steps(model, draft)
    logger.info(
        "monokernel qwen38_flash_next MTP draft: draft extend of R x %d rows and "
        "draft decode of R x 1, R <= %d, in one launch each",
        L, runner.max_reqs,
    )  # fmt: skip
    return draft


SPEC = MonoSpec(
    name="qwen38_flash_next_mtp_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, target.MAX_ROWS + 1)),
            tp_sizes=(1,),
            resident_blocks=BLOCKS,
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    # the launcher checks the target's requirements
    requires=Requirements(attention_backends=target.SPEC.requires.attention_backends),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # MONOKERNEL_API_VERSION of the spec module
)
