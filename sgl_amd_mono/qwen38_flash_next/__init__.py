# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next (SGLang's Qwen4-Exp) on gfx950: the target verify and the
MTP draft monokernels (RFC #43949).

``spec.py`` serves the target's speculative verify steps (the whole model in 13
launches), ``draft.py`` the MTP draft's steps (one launch each); the kernels and
their runner live in ``gfx950/``, the views of SGLang's weights in ``weights.py``.
"""

# The text config fields the kernels are built for (``gfx950/config.py``); the
# checkpoint nests them in ``text_config``
TEXT_CONFIG = {
    "hidden_size": 2560,
    "hc_count": 4,
    "hc_lowrank": 320,
    "rms_norm_eps": 1e-6,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "partial_rotary_factor": 0.25,
    "output_gate_type": "sigmoid",
    "full_attention_interval": 4,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "indexer_compress_ratio": 4,
    "indexer_budget": 2048,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ple_embed_dim": 2560,
    "ple_layer_ids": [2],  # 1-based: the PLE block sits on layer 1
    "mtp_num_hidden_layers": 1,
}
TARGET_ARCH = "Qwen4ExpForConditionalGeneration"
DRAFT_ARCH = "Qwen4ExpForCausalLMMTP"


def text_config(hf_config):
    """The text model's config: the VL checkpoint nests it."""
    return getattr(hf_config, "text_config", None) or hf_config


def text_config_reasons(hf_config, draft: bool = False) -> list[str]:
    """Why the kernels do not fit the config. SGLang rewrites the MTP draft's config
    to its one full-attention layer (``full_attention_interval=1``), so the draft is
    held to that layer instead of the target's stack."""
    t = text_config(hf_config)
    stack = {"full_attention_interval"}
    reasons = [
        f"config {k}={getattr(t, k, None)!r}, the kernels are built for {v!r}"
        for k, v in TEXT_CONFIG.items()
        if getattr(t, k, None) != v and not (draft and k in stack)
    ]
    kinds = list(getattr(t, "layer_types", None) or [])
    if draft:
        if kinds != ["full_attention"]:
            reasons.append(
                f"the MTP draft has layers {kinds}, not one full-attention layer"
            )
        return reasons
    want = [
        "full_attention" if i % 4 == 3 else "linear_attention"
        for i in range(len(kinds))
    ]
    if not kinds or len(kinds) % 4 or kinds != want:
        reasons.append(
            "the layers are not a repeating 3 x GDN + 1 x full-attention stack"
        )
    return reasons
