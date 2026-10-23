"""The Qwen3.8-Flash-Next servers that the monokernel serving test launches: one
MI355X GPU, NEXTN speculative decoding with the MTP draft (topk 1)."""

import os

MODEL = os.environ.get(
    "QWEN38_FLASH_NEXT_MODEL_PATH", "amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8"
)
# what the spec requires (SGLANG_USE_AITER, per-token FP8 activations, bf16 GDN
# states), AITER's unified attention and FlyDSL kernels, and the memory fraction
# honored as given
ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "AITER_FLYDSL_FORCE": "1",
    "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
    "SGLANG_USE_AITER_FP8_PER_TOKEN": "1",
    "SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION": "1",
}
# Up to 16 requests: a verify of 1 to 8 requests runs the launches, 9 to 16 the
# normal path, so a batch that fills and drains crosses the switch point. Without
# --monokernel-backend, so a test can launch the normal path too.
MAX_RUNNING = 16


def args(k: int) -> list[str]:
    """The server arguments at ``k`` draft steps (``k + 1`` tokens a verified request)."""
    return [
        "--tp-size",
        "1",
        "--attention-backend",
        "aiter",
        "--mem-fraction-static",
        "0.9",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        str(k),
        "--speculative-num-draft-tokens",
        str(k + 1),
        "--speculative-eagle-topk",
        "1",
        "--max-running-requests",
        str(MAX_RUNNING),
        "--cuda-graph-max-bs-decode",
        str(MAX_RUNNING),
        "--reasoning-parser",
        "qwen3",
    ]
