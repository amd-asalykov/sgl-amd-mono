# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next on gfx950: the target's monokernel spec and forward (RFC #43949).

SGLang's runtime switches the model's forward at attach: a routed speculative
verify step (R <= 8 requests x L = num_draft_tokens <= 4 rows, topk 1) runs
``_Built.forward``, which takes and returns what
``Qwen4ExpForConditionalGeneration.forward`` does. The whole model runs in 13
persistent launches (``gfx950/runner.py``): the embedding and the PLE n-gram
prelude with layer 0, layers 1-3 behind the PLE block, then three GDN and one
full-attention (QSA) layer per launch, the last one ending in the final mixer;
SGLang's logits processor closes the step. The launches write what SGLang's
verify writes for every row: the KV cache and the QSA index ring, compressed
keys and RoPE positions, the GDN, PLE and n-gram verify scratch the accept step
commits from, and the residual streams the MTP draft reads.

The kernels read SGLang's weights in place except the copies ``weights.py``
names; a verify of more than 8 requests runs SGLang's forward.

``SGLANG_AMD_MONO_CHECK=1`` runs SGLang's forward ahead of every eager routed
step, on the same inputs, and logs both outputs' differences on the step's rows
(the bounds in ``NUMERICS`` come from these runs).
"""

from __future__ import annotations

import logging
import os
import re

import msgspec
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
from sgl_amd_mono.qwen38_flash_next import (
    TARGET_ARCH,
    text_config,
    text_config_reasons,
)

logger = logging.getLogger(__name__)

MAX_ROWS = 32  # gfx950/layout.MAX_T: requests x tokens a launch takes
MAX_REQS = 8  # gfx950/layout.MAX_R
MAX_TOKENS = 4  # draft tokens a request (num_draft_tokens) the launches take
MAX_CTX = 294912 - 64  # gfx950/layout.MAX_CTX less the verify rows' headroom

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "batch_size",
        "key",
    ),
    "srt/arg_groups/fields/*.py": (
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
        "kv_cache_dtype",
        "qsa_indexer_dtype",
        "ple_offload_embedding",
        "disable_shared_experts_fusion",
        "expert_distribution_recorder_mode",
        "context_length",
        "model_path",
    ),
    "srt/environ.py": (
        "envs",
        "SGLANG_USE_AITER",
        "SGLANG_USE_AITER_FP8_PER_TOKEN",
        "SGLANG_MAMBA_SSM_DTYPE",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        "get_spec",
        "get_server_args",
        '"tp_size"',
        '"pp_size"',
    ),
    "srt/model_executor/forward_context.py": (
        "get_req_to_token_pool",
        "get_token_to_kv_pool",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "input_ids",
        "positions",
        "out_cache_loc",
        "req_pool_indices",
        "seq_lens",
        "spec_info",
    ),
    "srt/layers/logits_processor.py": (
        "LogitsProcessorOutput",
        "next_token_logits",
        "hidden_states",
    ),
    "srt/layers/rotary_embedding/base.py": ("cos_sin_cache_as",),
    # Qwen4ExpForConditionalGeneration.forward, which the routed forward stands in for
    "srt/models/qwen4_exp.py": (
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(get_embedding)",
        "forward(pp_proxy_tensors)",
        "last_hc_hidden_states",
        "ple",
    ),
    "srt/models/qwen3_5.py": ("start_layer", "end_layer", "rotary_emb", "layers"),
    "srt/models/qwen3_vl.py": ("logits_processor", "lm_head", "model"),
    # the pools the launches read and write (gfx950/runner.py)
    "srt/mem_cache/memory_pool.py": (
        "req_to_token",
        "req_index_to_mamba_index_mapping",
        "mamba2_layer_cache",
        "short_conv_layer_cache",
        "short_conv_layer_intermediate_cache",
        "intermediate_ssm",
        "intermediate_conv_window",
        "temporal",
        "conv",
        "ngram_pool",
        "get_key_buffer",
        "get_value_buffer",
    ),
    "srt/mem_cache/ple_state_pool.py": ("context", "intermediate_context"),
    "srt/mem_cache/qsa_kv_pool.py": (
        "qsa_rope_position_buffer",
        "get_qsa_key_state_buffer",
        "get_qsa_compressed_k_buffer",
    ),
}

# The launches against SGLang's forward on the same inputs (SGLANG_AMD_MONO_CHECK,
# eager verify steps of 1 to 8 requests at k=3 and k=1, about 1300 steps; worst values
# with a margin; cosine and rel over the rows of the step). Integer outputs are exact.
# The first launch block (layers 0-3) and the PLE block are tight; the 48 layers'
# outputs carry the compounded difference of FP8 quantization points and reduction
# order, and the deeper layers' regions are reported, not bounded. The compressed index
# keys are reported, not bounded: SGLang's verify pools a group completed before the
# step's last token with keys its later tokens wrote over in the ring.
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "verify.rope.max_abs": 0,
        "verify.ngram.max_abs": 0,
        "verify.first_gdn_conv.cosine": 0.995,
        "verify.first_gdn_ssm.cosine": 0.997,
        "verify.first_kv.cosine": 0.98,
        "verify.first_ring.cosine": 0.99,
        "verify.ple.cosine": 0.995,
        "verify.logits.cosine": 0.85,
        "verify.logits.rel": 0.6,
        "verify.hidden.cosine": 0.70,
        "verify.hidden.rel": 0.9,
    },
)

# checkpoint name -> (layer id, name below the layer); the final mixer sits at the top
_TARGET_NAME = re.compile(r"model\.language_model\.layers\.(?P<i>\d+)\.(?P<rest>.+)")


def _kinds(hf_config) -> tuple[str, ...]:
    return tuple(
        "qsa" if k == "full_attention" else "gdn"
        for k in text_config(hf_config).layer_types
    )


def _has_flydsl() -> bool:
    import importlib.util

    return importlib.util.find_spec("flydsl") is not None


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """What ``SPEC.requires`` cannot state: the nested text config, the verify
    shape, the context length and FlyDSL."""
    reasons = text_config_reasons(hf_config)
    if cfg.speculative_algorithm is None:
        reasons.append("the launches serve speculative verify steps only (NEXTN)")
    elif not 2 <= int(cfg.speculative_num_draft_tokens or 0) <= MAX_TOKENS:
        reasons.append(
            f"{cfg.speculative_num_draft_tokens} draft tokens a request; the launches "
            f"take 2..{MAX_TOKENS}"
        )
    t = text_config(hf_config)
    ctx = cfg.context_length or getattr(t, "max_position_embeddings", 0)
    if ctx > MAX_CTX:
        reasons.append(f"context length {ctx} past the launches' {MAX_CTX}")
    if not _has_flydsl():
        reasons.append("needs FlyDSL")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel, get_server_args

    from sgl_amd_mono.qwen38_flash_next import weights as W

    p = get_parallel()
    reasons = []
    if (p.tp_size, p.pp_size) != (1, 1):
        reasons.append("the launches run the whole model on one GPU: TP and PP 1")
    m = model.model
    kinds = _kinds(hf_config)
    if (m.start_layer, m.end_layer) != (0, len(kinds)):
        reasons.append(f"layers {m.start_layer}..{m.end_layer} of {len(kinds)}")
        return reasons
    # the copies come from the checkpoint files, not a load hook: every load path keeps them
    _read_copies(copies, model, get_server_args().model_path)
    unseen = W.target_unseen(copies, kinds)
    if unseen:
        reasons.append(
            f"{len(unseen)} checkpoint tensors the launches pack are not in the checkpoint "
            f"(first: {unseen[0]})"
        )
    for i, kind in enumerate(kinds):
        reasons += W.moe_reasons(m.layers[i].mlp.experts, f"layer {i}")
    reasons += W.ngram_reasons(m.layers[1].ple)
    if reasons:
        return reasons
    # SGLang's AITER layouts against the checkpoint, byte for byte
    for i in W.CHECK_LAYERS:
        moe = W.moe_params(m.layers[i].mlp.experts)
        reasons += W.check_experts(moe, copies, i, f"layer {i}")
        table = W.GDN_PROJ if kinds[i] == "gdn" else W.QSA_PROJ
        scratch: dict = {}
        proj, why = W.proj_params(m.layers[i], table, scratch, i)
        reasons += why or W.check_proj(proj, table, copies, i, f"layer {i}")
    return reasons


def _read_copies(copies: dict, model, path: str) -> None:
    """Keep what ``_keep_tensor`` keeps, from the checkpoint files under ``path``."""
    from sgl_amd_mono.qwen38_flash_next import weights as W

    W.read_checkpoint(copies, path, lambda c, name, t: _keep_tensor(c, model, name, t))


def _keep_tensor(copies: dict, model, name: str, tensor) -> None:
    """Keep the checkpoint tensors the launches pack, and the few a startup check
    compares with SGLang's (``weights.keep_checkpoint_tensor``), on the CPU."""
    from sgl_amd_mono.qwen38_flash_next import weights as W

    m = _TARGET_NAME.fullmatch(name)
    if m is not None:
        kinds = dict(enumerate(_kinds(model.config)))
        W.keep_checkpoint_tensor(copies, int(m["i"]), m["rest"], tensor, kinds)
    elif name.startswith(W.TARGET_PFX):
        W.keep_final(copies, name[len(W.TARGET_PFX) :], tensor)


class _Built:
    """The monokernels of the target: the runner and the forward."""

    def __init__(self, lm, runner):
        self.lm, self.runner = lm, runner
        self.rope = lm.model.layers[3].rotary_emb  # a full-attention layer's
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1"

    def forward(self, plan, input_ids, positions, forward_batch, get_embedding=False,
                pp_proxy_tensors=None, *args, **kwargs):  # fmt: skip
        """``Qwen4ExpForConditionalGeneration.forward`` for a routed verify step."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.forward_context import (
            get_req_to_token_pool,
            get_token_to_kv_pool,
        )

        lm = self.lm
        if get_embedding or plan.key.batch_size > self.runner.max_reqs:
            # more requests than a launch takes: SGLang's forward (graph-safe: by key)
            return type(lm).forward(lm, input_ids, positions, forward_batch,
                                    get_embedding, pp_proxy_tensors)  # fmt: skip
        rtp, kvp = get_req_to_token_pool(), get_token_to_kv_pool()
        check = None
        if self.check and not torch.cuda.is_current_stream_capturing():
            check = self._reference(plan, input_ids, positions, forward_batch, rtp, kvp)
        cos = self.rope.cos_sin_cache_as(torch.bfloat16)  # f32 on HIP in SGLang
        x, h = self.runner.forward(self._batch(forward_batch), rtp, kvp, cos)
        lm.model.last_hc_hidden_states = h
        out = lm.logits_processor(input_ids, x, lm.lm_head, forward_batch, None)
        if isinstance(out, LogitsProcessorOutput):
            out.hidden_states = h
        if check is not None:
            self._report(plan, out, *check)
        return out

    def _batch(self, forward_batch):
        """The step's batch, with the RoPE positions SGLang's forward applies: its
        MRoPE positions (``[3, rows]``) where the model enables them, which differ
        from the sequence positions in a request with images or video."""
        from sgl_amd_mono.qwen38_flash_next.gfx950.runner import StepBatch

        batch = StepBatch.of(forward_batch)
        if self.lm.is_mrope_enabled:
            batch = msgspec.structs.replace(
                batch, positions=forward_batch.mrope_positions
            )
        return batch

    def _reference(self, plan, input_ids, positions, forward_batch, rtp, kvp):
        """Check mode: SGLang's forward on the step's inputs; the regions it wrote
        are kept and put back as they were, for the launches."""
        from sgl_amd_mono.qwen38_flash_next import check as C

        lm, rows = self.lm, plan.key.rows
        regions = C.target_regions(self.runner.w.kinds, forward_batch, rows, rtp, kvp)
        pre = C.snapshot(regions)
        out = type(lm).forward(lm, input_ids, positions, forward_batch)
        ref = C.snapshot(regions)
        C.restore(regions, pre)
        logits = out.next_token_logits[:rows].float().clone()
        return regions, pre, ref, logits, out.hidden_states[:rows].float().clone()

    def _report(self, plan, out, regions, pre, ref, logits, hidden) -> None:
        from sgl_amd_mono.qwen38_flash_next import check as C

        rows = plan.key.rows
        values = C.compare("verify", regions, ref, pre, {
            "logits": (out.next_token_logits[:rows], logits),
            "hidden": (out.hidden_states[:rows], hidden),
        })  # fmt: skip
        agree = (
            (out.next_token_logits[:rows].argmax(-1) == logits.argmax(-1))
            .float()
            .mean()
            .item()
        )
        label = f"verify {plan.key.batch_size} x {plan.key.tokens_per_request} (argmax agrees on {agree:.3f})"
        C.report(label, values, NUMERICS)


def _build(model, hf_config, ctx) -> _Built:
    from sglang.srt.runtime_context import get_spec

    from sgl_amd_mono.qwen38_flash_next import weights as W
    from sgl_amd_mono.qwen38_flash_next.gfx950.runner import TargetRunner, TargetWeights

    copies = ctx.copies
    W.drop_checks(copies)
    m, t = model.model, text_config(hf_config)
    kinds = _kinds(hf_config)
    layers, ple = {}, None
    for i, kind in enumerate(kinds):
        layers[i], p = W.pack_target_layer(m.layers[i], i, kind, copies)
        ple = p if p is not None else ple
        torch.cuda.empty_cache()
    table, scale, meta = W.ngram_tables(m.layers[1].ple)
    weights = TargetWeights(
        kinds=kinds, layers=layers, ple=ple, final=W.pack_final(copies),
        embed_w=m.embed_tokens.weight, ngram_table=table, ngram_scale=scale,
        ngram_meta=meta, eos_token_id=int(t.eos_token_id),
    )  # fmt: skip
    L = int(get_spec().speculative_num_draft_tokens)
    runner = TargetRunner(weights, L, m.layers[0].mlp.experts.w13_weight.device)
    runner.set_deadline(ctx.deadline_s)
    # Widths count rows, so at L = 2 the runtime routes up to 16 requests; the forward
    # runs SGLang's for more than the launches take (graph-safe: by key).
    logger.info(
        "monokernel qwen38_flash_next verify: the launches run %s requests x %d tokens; "
        "a routed step of more requests runs SGLang's forward",
        f"1..{runner.max_reqs}", L,
    )  # fmt: skip
    # The launches compile at the first step of each width (their keys hold the
    # pools' strides, sized after this build); FlyDSL's disk cache keeps them.
    return _Built(model, runner)


SPEC = MonoSpec(
    name="qwen38_flash_next_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, MAX_ROWS + 1)),
            tp_sizes=(1,),
            resident_blocks=BLOCKS,
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        # the launches replace the attention; prefill writes the bf16 KV both keep
        attention_backends=frozenset({"aiter", "triton"}),
        server_args={
            "speculative_algorithm": frozenset({"EAGLE", "NEXTN"}),
            "speculative_eagle_topk": frozenset({1}),
            "kv_cache_dtype": frozenset({"auto", "bf16", "bfloat16"}),
            "qsa_indexer_dtype": frozenset({"auto", "bfloat16"}),
            # the launches read the n-gram table from the GPU
            "ple_offload_embedding": frozenset({None, False}),
            # the shared expert runs as the fused expert 512 of the MXFP4 MoE
            "disable_shared_experts_fusion": frozenset({False}),
            # the launches record no expert distribution
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        env={
            # the experts and FP8 projections in AITER's layouts
            "SGLANG_USE_AITER": True,
            # the launches quantize activations per token, as SGLang's FP8 linears then do
            "SGLANG_USE_AITER_FP8_PER_TOKEN": True,
            # the GDN states the launches were validated on
            "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # MONOKERNEL_API_VERSION of the spec module
    config_disable_reasons=_config_disable_reasons,
)

assert TARGET_ARCH  # registered in sgl_amd_mono.SPECS
