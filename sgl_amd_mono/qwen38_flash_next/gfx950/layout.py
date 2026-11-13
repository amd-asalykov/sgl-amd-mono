# SPDX-License-Identifier: Apache-2.0
"""Compile-time geometry and scratch (mailbox) layout of the Qwen3.8-Flash-Next
launch, kept free of FlyDSL so it is testable without a GPU.

A launch serves R requests x L verify tokens = S rows (S <= 32). Every activation
mailbox is row-major [S][n], row t = request t // L, draft step t % L."""

import dataclasses
import functools

from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, Shard, padded

from .config import (
    GDN_DK,
    GDN_DV,
    GDN_HK,
    GDN_HV,
    GDN_QKVZ,
    HC_DIM,
    HC_LOWRANK,
    HIDDEN,
    MOE_INTER,
    N_EXPERTS,
    NGRAM_ROWS,
    QSA_HQ,
    QSA_KV,
    TOPK,
    TP_SIZES,
    VOCAB,
)

assert (BLOCKS, THREADS, WAVES) == (256, 512, 8)  # one CTA per MI355X CU, co-resident
MOE_PAIR = TOPK + 1  # (row, expert) pairs per row: top-10 routed + the shared expert

# hyper-connection mix
MIX_KSPLIT = 10  # down GEMV split-K across CTAs (1024-wide K slices)
MIX_DOWN_RG = HC_LOWRANK // 16 + 1  # 20 row tiles + the combine-gate tile
MIX_DOWN_TASKS = MIX_DOWN_RG * MIX_KSPLIT  # 210
MIX_UP_TASKS = HIDDEN // 16  # 160 (each covers 16 outputs x 4 branches)
# sum-of-squares partials per (token, branch) published with a residual stream (one
# per 16-row group)
NPART = HIDDEN // 16

# GDN
IP_ROWS = GDN_QKVZ + 2 * GDN_HV  # 16480: [q k v | z | b | a]
IP_RG_MAIN = GDN_QKVZ // 16  # 1024 = 4 per CTA; the core tasks compute the b/a rows
MAX_CORE_SPLIT = 8  # the scratch's bound on core_split


def core_split(T, tp=1):
    """V-row splits per value head of the GDN core: 4 a TP rank for every T, so the
    core keeps 192 tasks at every TP (8 at TP 1 would need 384 CTAs)."""
    return 4 * tp


def max_core_split(tp=1):
    """The scratch's bound on core_split (its ssq / og_amax partials a row)."""
    return max(MAX_CORE_SPLIT, core_split(0, tp))


def core_geom(split, hv=GDN_HV):
    """(V rows per task, tasks, threads per V row, K elements per thread, xor offsets
    of a row reduce) of the GDN core at ``split`` over ``hv`` value heads."""
    rows = 128 // split
    rpt = 512 // rows
    kpt = 128 // rpt  # state words per thread
    red_offs = tuple(rpt >> k for k in range(1, rpt.bit_length()))
    return rows, hv * split, rpt, kpt, red_offs


OUT_RG = HIDDEN // 16  # 160

# MoE
SHARED_EXPERT = N_EXPERTS  # index of the shared expert in the stacked arrays
ROUTER_ROWS = N_EXPERTS + 16  # 512 routed logits + shared gate + 15 zero rows
ROUTER_RG = ROUTER_ROWS // 16  # 33
# MXFP4 experts in SGLang's AITER layout. get_moe_weight_sizes(is_aiter_moe) pads the
# intermediate dim to the AITER alignment (w2 row 320 -> 384 bytes): FusedMoE w13
# [513, 2 * 768, 1280] (gate rows 0..639, zero rows to 767, up rows 768..1407, zero
# rows to 1535), w2 [513, 2560, 384] (K 640..767 zero); process_weights_after_loading
# applies shuffle_weight(w, (16, 16)) and e8m0_shuffles the scales over the flattened
# [E * rows, groups] (groups padded to 8). The kernel reads only the real rows / K
# (gate tiles 0..39, up tiles 48..87, down K chunks 0..4).
MOE_INTER_PAD = 768
# MXFP4 gate/up tasks per expert (40): gate tile n + up tile n + UP_TILE0
GU_PAIRS = MOE_INTER // 16
UP_TILE0 = MOE_INTER_PAD // 16  # 48: first up row tile of w13
DN_RG = HIDDEN // 16  # 160
GU_BYTES = 2 * MOE_INTER_PAD * HIDDEN // 2  # fp4 bytes per expert (w13)
GU_SGROUPS = HIDDEN // 32  # 80
GU_SBYTES = 2 * MOE_INTER_PAD * GU_SGROUPS
DN_BYTES = HIDDEN * MOE_INTER_PAD // 2  # w2: 384-byte rows
DN_SGROUPS = (MOE_INTER_PAD // 32 + 7) // 8 * 8  # 24
DN_SBYTES = HIDDEN * DN_SGROUPS


def max_slots(T):
    """Union of routed experts of T tokens + the shared expert."""
    return min(TOPK * T, N_EXPERTS) + 1


def max_groups(T, sg):
    """MoE down K-groups of ``sg`` expert slots (partials summed by the combine)."""
    return (max_slots(T) + sg - 1) // sg


def kphases(T):
    """K phases of the K=2560 activation GEMMs: [T][2560] bf16 rows fit the LDS for
    T <= 16; above, the CTA stages the activation in two K halves (accumulators of
    all its row tiles kept across the halves)."""
    return 1 if T <= 16 else 2


def gu_kphases(T):
    """K halves of the MoE gate/up: the staged [T][2560] bf16 rows + the route tables
    fit the LDS for T <= 28; above, a CTA stages one K half and publishes f32 gate/up
    partials (summed by the down staging)."""
    return 1 if T <= 28 else 2


def gu_wsplit(T, tp=1):
    """Waves sharing one MoE gate/up task (an in-CTA K split, partials summed in LDS).
    TP > 1 at T <= 4: the rank's (slot, tile) tasks (at most 41 x 20) fill only 820 of
    the 2048 waves, so two waves take K halves (one round of 20 units a wave instead
    of 40). TP 1 (1640 tasks) and larger T: 1 (wave-level tasks)."""
    return 2 if tp > 1 and T <= 4 and gu_kphases(T) == 1 else 1


def dn_sg(T):
    """Expert slots per MoE down K-group: the staged [T][SG * 640] bf16 mid of a
    group must fit the LDS."""
    return 4 if T <= 28 else 3


# QSA (full attention) layer
# [q | gate per head | k | v | index q (4 x 128) + index k (128)]
QP_ROWS = 24 * 512 + 512 + 512 + 640
QP_RG = QP_ROWS // 16  # 872
QP_K_OFF = 24 * 512
QP_V_OFF = QP_K_OFF + 512
QP_IDX_OFF = QP_V_OFF + 512
# top-512 selection: an exact top-512 per segment (16 blocks a thread), then a merge
SEG_BLOCKS = 8192
NSEG_MAX = 9
MAX_BLOCKS = SEG_BLOCKS * NSEG_MAX  # 73728 compressed blocks
MAX_CTX = MAX_BLOCKS * 4  # 294912 >= max_position_embeddings (262144) + verify tokens
ATT_CHUNK = 64  # tokens per attention split
ATT_CHUNKS = 2048 // ATT_CHUNK + 1  # 32 block chunks + 1 tail chunk
ATT_TASKS = ATT_CHUNKS  # per token; one task covers both kv heads
ATT_BLK_TASKS = ATT_CHUNKS - 1  # the block chunks (stage qsa_attn); the tail chunk runs in qsa_tail
PART_STRIDE = 24 * 256 + 2 * 24  # per task: o[24][256], m[24], l[24]
MERGE_TASKS = 24  # one per (token, head)
PREP_Q_TASKS = 24  # CTAs [0, 24): one attention q head each (all tokens)
PREP_K_CTA = 24  # CTAs 24, 25: k head (norm+rope) + v -> KV cache
# CTAs [26, 26 + R): index q + index k ring / compress of request r (its tokens, in
# order)
PREP_IDX_CTA0 = 26


# Stage placement: task k of a placed stage runs on CTA (base + k) % 256, so the CTAs
# a stage leaves idle issue the next stage's weights while it runs (every CTA still
# walks the stages in order: deadlock-free, outputs unchanged).
MIXA_BASE = DN_RG  # mix-a down past the MoE combine's CTAs [0, 160)


def op_base(kind, T, tp=1):
    """out_proj / o_proj base: past the GDN core's CTAs [0, C_TASKS) (192) for T > 4
    (at T = 4, CTA 0 measured faster), or past the QSA merge's last round."""
    g = geom(tp)
    if kind == "gdn":
        return g.HV * core_split(T, tp) if T > 4 else 0
    return (T * g.MERGE_TASKS) % 256


def mixm_base(kind, T, tp=1):
    """mix-m down base: past out_proj."""
    return (op_base(kind, T, tp) + OUT_RG) % 256


# QSA prep tasks start at CTA PREP_BASE (past the CTAs [0, 104) with a 4th qsa_proj
# tile: 872 = 3 x 256 + 104), the index scores right after them
PREP_BASE = QP_RG % 256
SCORES_BASE = PREP_BASE + PREP_IDX_CTA0 + 8  # + MAX_R


def sel2_cta0(T):
    """CTAs [256 - T, 256): merge of the segment top-512s of row t (long contexts)."""
    return 256 - T


def _tail_ctas(base, T):
    return {(base + t) % 256 for t in range(T)}


def scores_ctas(R):
    """CTAs of the index scores at an ~8k context: 2048 + 1 compressed blocks -> 129 tiles of 16 per request,
    8 tiles per CTA (the 129th, the newest block, gates the select of the request's last rows)."""
    return (R * (2048 // 16 + 1) + 7) // 8


def tail_base(R, T, tp=1):
    """QSA attention tail chunk of row t on CTA tail_base + t (stage qsa_tail): past the index scores' CTAs, or,
    if that reaches the select CTAs, just below the prep CTAs (at TP ``tp``: the rank's prep / scores bases).
    Placement only: the tail polls prep outputs alone, so any CTA range is deadlock-free."""
    g = geom(tp)
    base = (g.SCORES_BASE + scores_ctas(R)) % 256
    if _tail_ctas(base, T) & set(range(sel2_cta0(T), 256)):
        base = g.PREP_BASE - T
    return base


def _check_tail_base():
    for tp in TP_SIZES:
        g = geom(tp)
        for R in range(1, 9):
            for L in range(1, 32 // R + 1):
                T = R * L
                tail = _tail_ctas(tail_base(R, T, tp), T)
                prep = {(g.PREP_BASE + k) % 256 for k in range(g.PREP_IDX_CTA0 + R)}
                scores = {(g.SCORES_BASE + k) % 256 for k in range(scores_ctas(R))}
                sel = set(range(sel2_cta0(T), 256))
                assert not tail & (prep | scores | sel), (tp, R, L)


SEL_STRIDE = 1026  # pairs per token: (block, loc) as 2 pairs each, count at pair 1024


def cnew_slots(L):
    """Compressed blocks completed inside one request's L verify tokens."""
    return L // 4 + 2


# per-layer pointer table slots
P_HCA_NORM, P_HCA_DOWN, P_HCA_UP, P_HCA_INJ = 0, 1, 2, 3
P_IP, P_CONV_W, P_NEG_A, P_DT_BIAS, P_NORM_W, P_OUT = 4, 5, 6, 7, 8, 9
P_CONV_STATE, P_SSM_STATE = 10, 11
P_HCM_NORM, P_HCM_DOWN, P_HCM_UP, P_HCM_INJ = 12, 13, 14, 15
P_ROUTER, P_GU, P_GU_S, P_DN, P_DN_S = 16, 17, 18, 19, 20
# QSA reuses P_IP (fused q/k/v/index projection) and P_OUT (o_proj)
P_QNORM, P_KNORM, P_IQNORM, P_IKNORM = 5, 6, 7, 8
P_KBUF, P_VBUF, P_RING, P_COMP = 10, 11, 21, 22
P_BA = 23  # GDN: row-major [b (48) ; a (48)] x HIDDEN bf16
# PLE (layer 1 only): fused into the start of the launch whose first layer carries it
P_PLE_W, P_PLE_NK, P_PLE_NQ, P_PLE_NC, P_PLE_CW, P_PLE_STATE = 24, 25, 26, 27, 28, 29
# GDN verify scratch: per-draft-step SSM states / conv windows (batch row 0)
P_ISSM, P_ICONV = 30, 31
# PLE verify scratch: per-draft-step short-conv windows (GDN rows leave P_RING free)
P_PLE_ISTATE = 21
PLE_ROWS = HC_DIM + HIDDEN  # [key_proj (10240) ; value_proj (2560)]
PLE_RG = PLE_ROWS // 16  # 800
PLE_STATE_LEN = 9
PLE_STATE_SLOT_BYTES = HC_DIM * PLE_STATE_LEN * 2
PLE_CH_PER_CTA = HC_DIM // 256  # 40
PLE_PARTS = HIDDEN // PLE_CH_PER_CTA  # 64 ple_h producers per branch
N_PTR = 40
# FP8 projections: F32 per-output-channel weight scales
P_IP_S, P_OUT_S, P_BA_S = 32, 33, 34
# QSA: SGLang keeps qkv_proj and indexer.index_qk_proj as separate FP8 params (rows
# [0, 832 * 16) / the rest of QP)
P_IP2, P_IP2_S = 35, 36
QP_RG_QKV = (24 * 512 + 512 + 512) // 16  # 832
BA_ROWS = 128  # SGLang pads in_proj_ba (96 rows) to 128 for AITER's CK GEMM
# global row (index n_layers of the table)
G_POS, G_LOC, G_REQ, G_SEQ = 0, 1, 2, 3
G_RTT, G_COS, G_ROPEPOS, G_MAMBA_MAP = 4, 5, 6, 7
# TP > 1: rows of this rank's vocab shard of the token embedding (SGLang's
# VocabParallelEmbedding: rank r holds rows [r * n, (r + 1) * n))
G_VOCAB_ROWS = 8
G_PLE_EMB = 9
G_FIN_OUT = 10  # final mixer output x [T, HIDDEN] bf16 (final_mix launches)
G_HIN_SSQ, G_HOUT_SSQ = 11, 12  # f32 [T, 4, NPART] sum-of-squares of h_in / h_out
G_INPUT_IDS, G_EMBED_W = 13, 14
G_NGRAM_W, G_NGRAM_SCALE, G_NGRAM_CTX, G_NGRAM_META = 15, 16, 17, 18
# int64 [T, 2] per-draft-step n-gram history (batch row 0 of intermediate_context)
G_NGRAM_ICTX = 19
# MTP draft launches (first_src "mtp"): input fusion of the draft model
# bf16 [T, HC_DIM] target (or previous draft step) hyper-connection hidden states
G_MTP_HIN = 20
# pre_fc_norm_embedding [HIDDEN], pre_fc_norm_hidden [HC_DIM] (bf16)
G_MTP_PE, G_MTP_PH = 21, 22
G_MTP_FCE, G_MTP_FCH = 23, 24  # packed bf16 fc_embedding / fc_hidden [HIDDEN, HIDDEN]
# TP > 1: rows of this rank's vocab shard of the PLE n-gram table (from global row
# rank * n)
G_NGRAM_ROWS = 25
NGRAM_HEADS, NGRAM_DIM, HEADS_PER_NGRAM = 16, 160, 8
PRELUDE_CTA0 = 224  # CTAs [224, 224 + T): PLE n-gram rows of verify row t
CONV_SLOT_BYTES = 10240 * 3 * 2  # bf16 [10240, 3] per mamba slot
SSM_SLOT_BYTES = 48 * 128 * 128 * 4  # f32 [48, 128, 128] per mamba slot
SSM_STEP_WORDS = 48 * 128 * 128
LAUNCH_SLOTS = 64  # tag = step * LAUNCH_SLOTS + launch_id + 1


def scratch_layout(n_layers: int, T: int, R: int = 1, tp: int = 1):
    """Byte offsets of every tagged-pair mailbox, one region per (layer, stage), and
    the total size. T = rows, R = requests, tp: the rank's shard geometry (the
    replicated stages' regions do not depend on it)."""
    g = geom(tp)
    regions = {}
    off = 0

    def add(name, pairs):
        nonlocal off
        regions[name] = off
        off += padded(pairs * 8, 256)

    for layer in range(n_layers):
        p = f"L{layer}."
        add(p + "qp", T * g.QP_ROWS // 2)
        add(p + "q", T * g.HQ * 256 // 2)
        add(p + "kvnew", T * g.KV * 512 // 2)  # k 256 + v 256 a local kv head
        add(p + "qi", T * 512 // 2)
        add(p + "cnew", R * cnew_slots(T // R) * 64)  # [R][CN][64]
        add(p + "ilog", T * MAX_BLOCKS)
        add(p + "sel", T * SEL_STRIDE)
        # per (token, segment): key bits of the segment's top-512
        add(p + "cand_k", T * NSEG_MAX * 512)
        add(p + "cand_bl", T * NSEG_MAX * 512 * 2)  # (block, KV loc)
        add(p + "part", T * ATT_TASKS * g.PART_STRIDE)
        add(p + "og", T * g.HQ * 256 // 2)
        add(p + "ple_kv", T * PLE_ROWS // 2)
        add(p + "ple_h", T * HC_DIM // 2)
        add(p + "ple_ssq", T * 4 * PLE_PARTS)
        for hc in ("a", "m"):
            add(p + hc + ".mixd", MIX_KSPLIT * T * HC_LOWRANK)
            add(p + hc + ".mixx", T * HIDDEN // 2)  # bf16 pairs
            add(p + hc + ".injp", MIX_KSPLIT * T * 4)
        add(p + "ip", T * g.IP_ROWS // 2)  # bf16 pairs
        # per (token, head, split) sum of squares of the GDN output
        add(p + "ssq", T * g.HV * g.MAX_CORE_SPLIT)
        # per (row, producer task) max |og| (the FP8 activation scale); a GDN core task
        # publishes (ssq, max |og|) as 2 pairs when out_proj applies the gated norm
        # (S <= 8): 2 pairs a core task (TP 1: HV x MAX_CORE_SPLIT = 2 x 192 too)
        add(p + "og_amax", T * max(g.HV * g.MAX_CORE_SPLIT, 2 * g.HV * core_split(T, tp)))
        add(p + "h1", T * HC_DIM // 2)  # bf16 pairs
        add(p + "h1ssq", T * 4 * NPART)
        add(p + "logit", T * ROUTER_ROWS)  # f32
        # (row, expert) pairs p = t * 11 + rank: bf16 h = silu(g) * u [p][I], and
        # under gu_kphases 2 the f32 K-half-0 gate/up partials [p][2 I]
        add(p + "mid", MOE_PAIR * T * g.MOE_INTER // 2)
        if gu_kphases(T) == 2:
            add(p + "gupart", MOE_PAIR * T * 2 * g.MOE_INTER)
        add(p + "dpart", max_groups(T, dn_sg(T)) * T * HIDDEN)  # f32 [group][T][2560]
        add(p + "h2", T * HC_DIM // 2)  # bf16 pairs
        add(p + "h2ssq", T * 4 * NPART)
        # MTP input fusion output (layer-0 input of a draft launch)
        add(p + "mtp_h", T * HC_DIM // 2)
        add(p + "mtp_ssq", T * 4 * NPART)
    return regions, off


# ---- debug timeline (QWEN38_TIMELINE=1 at build): thread 0 of each CTA stores the
# 100 MHz clock after every stage into a [CTA][TL_POINTS] int64 row past the mailboxes
TL_POINTS = 512


def timeline_enabled() -> bool:
    import os

    return os.environ.get("QWEN38_TIMELINE", "0") == "1"


def timeline_offset() -> int:
    """Byte offset of the timeline rows: past every runner's mailboxes (at any TP:
    the per-rank geometry only shrinks them)."""
    verify = max(scratch_layout(4 + 2, r * 4, r)[1] for r in range(1, 9))
    draft = max(scratch_layout(1 + 2, r * t, r)[1] for r in range(1, 9) for t in (1, 4))
    return padded(max(verify, draft), 4096)


def timeline_bytes(blocks: int) -> int:
    return blocks * TL_POINTS * 8


# ---- per-rank geometry at TP (``notes/TP_PLAN.md`` section 1): SGLang's TP shards
@dataclasses.dataclass(frozen=True)
class Geom:
    """The shapes one TP rank holds, laid out as SGLang shards them (attention TP =
    MoE TP = tp, EP 1): GDN heads, QSA q heads and KV head, and the MoE
    intermediate are split; the hyper-connection mixers, router, QSA indexer and
    PLE projections are replicated (their constants are the module-level ones).
    The module-level constants of the sharded quantities are ``geom(1)``'s."""

    tp: int
    # Gated DeltaNet: contiguous k / v head ranges (local v j -> local k j // 3)
    HK: int
    HV: int
    KEY_DIM: int
    VAL_DIM: int  # also out_proj's (and the QSA o_proj's) K
    CONV_DIM: int  # [q_r | k_r | v_r] channels
    QKVZ: int  # in_proj_qkvz rows [q_r | k_r | v_r | z_r]
    IP_ROWS: int  # + b / a rows
    IP_RG_MAIN: int
    BA_REAL: int  # in_proj_ba rows (b then a)
    BA_ROWS: int  # the kernel's zero-padded, shuffled copy
    MAX_CORE_SPLIT: int
    CONV_SLOT_BYTES: int
    SSM_SLOT_BYTES: int
    SSM_STEP_WORDS: int
    # QSA: q heads [r HQ, (r + 1) HQ); KV heads (TP 4: head r // 2, replicated in pairs)
    HQ: int
    KV: int
    QP_RG_QKV: int  # qkv_proj rows / 16: [HQ x (q 256 | gate 256) | k 256 KV | v 256 KV]
    QP_ROWS: int  # + the replicated indexer's 640 rows
    QP_RG: int
    QP_K_OFF: int
    QP_V_OFF: int
    QP_IDX_OFF: int
    PART_STRIDE: int
    MERGE_TASKS: int
    PREP_Q_TASKS: int
    PREP_K_CTA: int
    PREP_IDX_CTA0: int
    PREP_BASE: int
    SCORES_BASE: int
    # MoE: every expert (and the shared one) at intermediate I / tp, AITER-padded
    MOE_INTER: int
    MOE_INTER_PAD: int
    GU_PAIRS: int
    UP_TILE0: int
    GU_BYTES: int
    GU_SBYTES: int
    DN_BYTES: int
    DN_SGROUPS: int
    DN_SBYTES: int
    # vocab-parallel tables
    VOCAB: int
    NGRAM_ROWS: int


@functools.cache
def geom(tp: int = 1) -> Geom:
    """Rank geometry at TP ``tp`` (``config.TP_SIZES``)."""
    if tp not in TP_SIZES:
        raise ValueError(f"TP {tp}: the geometry is laid out for {TP_SIZES}")
    sh = Shard(tp)
    hk, hv = sh.split(GDN_HK, "GDN key heads"), sh.split(GDN_HV, "GDN value heads")
    key_dim, val_dim = hk * GDN_DK, hv * GDN_DV
    qkvz = 2 * key_dim + 2 * val_dim
    hq = sh.split(QSA_HQ, "QSA q heads")
    kv = max(1, QSA_KV // tp)  # SGLang's get_num_kv_heads
    qp_qkv = hq * 512 + 2 * 256 * kv
    qp_rows = qp_qkv + 640
    prep_k = hq
    prep_idx = prep_k + kv
    prep_base = qp_rows // 16 % 256
    inter = sh.split(MOE_INTER, "MoE intermediate")
    # get_moe_weight_sizes(is_aiter_moe): the packed w2 K aligned to 128 bytes
    ip = padded(inter, 256)
    dn_sg = (ip // 32 + 7) // 8 * 8
    return Geom(
        tp=tp,
        HK=hk,
        HV=hv,
        KEY_DIM=key_dim,
        VAL_DIM=val_dim,
        CONV_DIM=2 * key_dim + val_dim,
        QKVZ=qkvz,
        IP_ROWS=qkvz + 2 * hv,
        IP_RG_MAIN=qkvz // 16,
        BA_REAL=2 * hv,
        BA_ROWS=padded(2 * hv, 64),
        MAX_CORE_SPLIT=max_core_split(tp),
        CONV_SLOT_BYTES=(2 * key_dim + val_dim) * 3 * 2,
        SSM_SLOT_BYTES=hv * GDN_DK * GDN_DV * 4,
        SSM_STEP_WORDS=hv * GDN_DK * GDN_DV,
        HQ=hq,
        KV=kv,
        QP_RG_QKV=qp_qkv // 16,
        QP_ROWS=qp_rows,
        QP_RG=qp_rows // 16,
        QP_K_OFF=hq * 512,
        QP_V_OFF=hq * 512 + 256 * kv,
        QP_IDX_OFF=hq * 512 + 512 * kv,
        PART_STRIDE=hq * 256 + 2 * hq,
        MERGE_TASKS=hq,
        PREP_Q_TASKS=hq,
        PREP_K_CTA=prep_k,
        PREP_IDX_CTA0=prep_idx,
        PREP_BASE=prep_base,
        SCORES_BASE=prep_base + prep_idx + 8,
        MOE_INTER=inter,
        MOE_INTER_PAD=ip,
        GU_PAIRS=inter // 16,
        UP_TILE0=ip // 16,
        GU_BYTES=2 * ip * HIDDEN // 2,
        GU_SBYTES=2 * ip * GU_SGROUPS,
        DN_BYTES=HIDDEN * ip // 2,
        DN_SGROUPS=dn_sg,
        DN_SBYTES=HIDDEN * dn_sg,
        VOCAB=sh.split(VOCAB, "vocab"),
        NGRAM_ROWS=sh.split(NGRAM_ROWS, "n-gram table rows"),
    )


# ---- the symmetric peer buffer at TP > 1 (``notes/TP_PLAN.md`` decision 3, commit 2)
PEER_REGIONS = ("A", "F", "X")  # attention AR (out_proj / o_proj), MoE AR (down), prelude
PEER_ALIGN = 4096
# region X (the prelude / MTP vocab-shard exchange): a row t of [X_ROW_PAIRS tagged
# pairs]: the token embedding (X_EMB_PAIRS bf16 pairs), then the 16 PLE n-gram heads'
# FP8 rows (NGRAM_DIM / 4 words each), each pushed by the rank owning it
X_EMB_PAIRS = 2560 // 2
X_ROW_PAIRS = X_EMB_PAIRS + 16 * 160 // 4  # 1920
PEER_PAIRS_A_TILE = 8  # bf16 pairs a (row group, row) tile line: 16 columns
LAUNCHES_PER_VERIFY = 13  # odd: the running launch parity alternates across steps


def peer_layout(tp: int, S: int) -> dict:
    """Byte offsets of the symmetric peer buffer of a runner of at most ``S`` rows at
    TP ``tp``: one region per exchange kind of a launch, so no two exchanges of one
    launch (they carry the same tag) share a word, each in two parity halves (a fast
    rank's next launch writes the other half while a slow rank still reads this
    one), then the guard's error flags.

    A region-parity is [src rank W][row group 160][row S][8 tagged bf16 pairs]:
    rank ``src`` pushes its 16-column partial of row group ``rg`` into slice ``src``
    of every rank's buffer (its own too), tile-major, so the S rows of one tile are
    S whole 64 B pair runs (a wave's push fills whole 128 B lines); every rank sums
    the W slices of its own buffer in rank order. Region X carries a row's owner
    push (the vocab-sharded embedding and n-gram rows), laid out alike.

    -> {"A", "F", "X": region base, "_parity": parity stride, "_src": source slice
    stride, "flags", "_bytes"}."""
    from sgl_amd_mono.common.guard import FLAG_BYTES, FLAG_WORDS

    if tp not in TP_SIZES or not 1 <= S <= 32:
        raise ValueError(f"peer layout of TP {tp}, {S} rows")
    assert tp <= FLAG_WORDS
    src = OUT_RG * S * PEER_PAIRS_A_TILE * 8  # = S * HIDDEN * 4 bytes
    part = padded(tp * src, PEER_ALIGN)
    assert tp == 1 or S * X_ROW_PAIRS * 8 <= part, "region X rows"
    out, off = {}, 0
    for name in PEER_REGIONS:
        out[name] = off
        off += 2 * part
    out.update({"_parity": part, "_src": src, "flags": off, "_bytes": off + FLAG_BYTES})
    return out


def peer_pair(src: int, rg: int, t: int, p: int, S: int) -> int:
    """Pair index (8 B units) of bf16 pair ``p`` (columns 16 rg + 2 p, + 1) of row ``t``
    pushed by rank ``src``, inside a region-parity of ``peer_layout``."""
    return ((src * OUT_RG + rg) * S + t) * PEER_PAIRS_A_TILE + p


def verify_parity(step: int, launch_id: int) -> int:
    """The peer-buffer half of a verify launch: ``step * 13 + launch_id`` (the
    model's 13 launches a step, so it alternates across steps too; the tag's parity
    does not: launch 12 and the next step's launch 0 share it). With n launches a
    step, the last and the next step's launch 0 differ in parity iff n is odd
    (``check_parity_launches``; the runner refuses an even count at TP > 1). The
    halves are a second safeguard: consecutive launches never race on a region
    anyway (a launch ends on F, the next starts on A, and a rank's push into a
    region waits on its combine of the other, which needs every peer past its reads
    of the first)."""
    return (step * LAUNCHES_PER_VERIFY + launch_id) & 1


def check_parity_launches(n_launches: int, tp: int) -> None:
    """``verify_parity`` alternates across a step boundary only with an odd launch
    count a step (13 at 48 layers; test models: 8 layers -> 3, 4 -> 2 refused)."""
    if tp > 1 and n_launches % 2 == 0:
        raise ValueError(
            f"Qwen3.8-Flash-Next monokernel: TP {tp} needs an odd launch count a "
            f"verify step (the peer-buffer parity), got {n_launches}"
        )


def draft_parity(step: int) -> int:
    """The peer-buffer half of a draft launch (one launch a step)."""
    return step & 1


_check_tail_base()
