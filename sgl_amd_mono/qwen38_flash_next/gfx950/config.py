# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next's dimensions the gfx950 kernels are built for (the
checkpoint's ``text_config``; ``qwen38_flash_next.HF_CONFIG`` checks them)."""

HIDDEN = 2560
HC = 4  # hyper-connection streams
HC_DIM = HIDDEN * HC  # 10240, the residual stream width
HC_LOWRANK = 320
EPS = 1e-6
# Gated DeltaNet
GDN_HK = 16
GDN_HV = 48
GDN_DK = 128
GDN_DV = 128
GDN_KEY_DIM = GDN_HK * GDN_DK  # 2048
GDN_VAL_DIM = GDN_HV * GDN_DV  # 6144
GDN_CONV_DIM = 2 * GDN_KEY_DIM + GDN_VAL_DIM  # 10240
GDN_QKVZ = 2 * GDN_KEY_DIM + 2 * GDN_VAL_DIM  # 16384
GDN_CONV_K = 4
# MoE: 512 routed experts, top-10, and the shared expert fused at index 512
N_EXPERTS = 512
TOPK = 10
MOE_INTER = 640
# full attention (QSA)
ATT_HQ = 24
ATT_HKV = 2
ATT_D = 256
ROTARY_DIM = 64
