"""The Qwen3.8-Flash-Next per-rank TP geometry, on CPU (no GPU, no FlyDSL).

``layout.geom(tp)`` must reproduce SGLang's per-rank shapes (notes/
tp_sglang_sharding.md section 1: MergedColumnParallel / RowParallel / QKVParallel
splits, the AITER-padded MoE intermediate chunk, the vocab shards), and at TP 1 the
module-level constants the TP 1 kernel was built on. The weight checks
(``weights.layer_reasons``) must accept a rank's params at its TP and refuse the
params of another TP. Fake params live on the meta device (shapes only)."""

import dataclasses

import pytest
import torch

from sgl_amd_mono.qwen38_flash_next.gfx950 import config as C
from sgl_amd_mono.qwen38_flash_next.gfx950 import layout as LY
from sgl_amd_mono.qwen38_flash_next.gfx950 import packing
from sgl_amd_mono.qwen38_flash_next.gfx950 import weights as W

HIDDEN = C.HIDDEN
META = "meta"

# SGLang's per-rank shapes (tp_sglang_sharding.md section 1), by TP
SGLANG = {
    1: dict(qkvz=16384, ba=96, out_k=6144, conv=10240, hv=48, qkv=13312, kv=2,
            w13=1536, w2=384, w13s=1536, w2s=24, vocab=248320, ngram=320001536),
    2: dict(qkvz=8192, ba=48, out_k=3072, conv=5120, hv=24, qkv=6656, kv=1,
            w13=1024, w2=256, w13s=1024, w2s=16, vocab=124160, ngram=160000768),
    4: dict(qkvz=4096, ba=24, out_k=1536, conv=2560, hv=12, qkv=3584, kv=1,
            w13=512, w2=128, w13s=512, w2s=8, vocab=62080, ngram=80000384),
}  # fmt: skip


def test_geom_tp1_is_the_module_constants():
    # the TP 1 kernel was built on the module constants: geom(1) must not move one
    g = LY.geom(1)
    names = {f.name for f in dataclasses.fields(g)} - {"tp", "HQ", "KV", "BA_REAL"}
    for n in names:
        mod = getattr(LY, n, None)
        if mod is None:
            mod = getattr(C, "GDN_" + n, None) or getattr(C, n)
        assert getattr(g, n) == mod, n
    assert (g.HQ, g.KV, g.BA_REAL) == (24, 2, 96)
    assert LY.core_split(4) == LY.core_split(32) == 4
    for kind in ("gdn", "qsa"):
        for T in (4, 8, 32):
            assert LY.op_base(kind, T, 1) == LY.op_base(kind, T)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_geom_matches_sglang_per_rank_shapes(tp):
    g, s = LY.geom(tp), SGLANG[tp]
    assert g.QKVZ == s["qkvz"] and g.BA_REAL == s["ba"] and g.BA_ROWS % 64 == 0
    assert g.BA_ROWS >= g.BA_REAL
    assert g.VAL_DIM == s["out_k"] == g.HQ * 256  # GDN out_proj K = QSA o_proj K
    assert g.CONV_DIM == s["conv"] and g.HV == s["hv"] and g.HK * 3 == g.HV
    assert g.QP_RG_QKV * 16 == s["qkv"] and g.KV == s["kv"]
    assert g.QP_ROWS == s["qkv"] + 640  # + the replicated indexer
    assert 2 * g.MOE_INTER_PAD == s["w13"] and g.MOE_INTER_PAD // 2 == s["w2"]
    assert g.DN_SGROUPS == s["w2s"]
    assert g.VOCAB == s["vocab"] and g.NGRAM_ROWS == s["ngram"]
    # the plan's derived quantities (TP_PLAN.md section 1)
    assert g.IP_RG_MAIN == {1: 1024, 2: 512, 4: 256}[tp]
    assert g.QP_RG == {1: 872, 2: 456, 4: 264}[tp]
    assert g.GU_PAIRS == {1: 40, 2: 20, 4: 10}[tp]
    assert g.UP_TILE0 == {1: 48, 2: 32, 4: 16}[tp]
    assert (g.HQ, g.MERGE_TASKS, g.PREP_K_CTA) == (24 // tp,) * 3
    assert g.PART_STRIDE == 256 * g.HQ + 2 * g.HQ
    # the GDN core keeps 192 tasks at every TP
    split = LY.core_split(8, tp)
    rows, tasks, rpt, kpt, _ = LY.core_geom(split, g.HV)
    assert tasks == 192 and split <= g.MAX_CORE_SPLIT and rows * split == 128
    assert rpt * rows == 512 and kpt * rpt == 128
    assert g.SSM_SLOT_BYTES == g.HV * 128 * 128 * 4
    assert g.CONV_SLOT_BYTES == g.CONV_DIM * 3 * 2


@pytest.mark.parametrize("tp", [2, 4])
def test_moe_padding_matches_fusedmoe_at_tp(tp):
    pytest.importorskip("sglang")
    try:
        from sglang.srt.layers.moe.utils import get_moe_weight_sizes
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"SGLang's MoE utils do not import: {e}")
    g = LY.geom(tp)
    w13_dim, w2_dim = get_moe_weight_sizes(
        g.MOE_INTER, is_concat=True, is_packed=True, is_aiter_moe=True
    )[:2]
    assert (w13_dim, w2_dim) == (2 * g.MOE_INTER_PAD, g.MOE_INTER_PAD // 2)


def test_bad_tp_is_refused():
    for tp in (0, 3, 8):
        with pytest.raises(ValueError):
            LY.geom(tp)
    with pytest.raises(ValueError):
        C.MonoBuild(plan="gdn", nreq=1, ntok=4, tp=3)
    assert C.MonoBuild(plan="gdn", nreq=1, ntok=4).tp == 1


@pytest.mark.parametrize("T,R", [(4, 1), (8, 2), (32, 8)])
def test_scratch_layout_tp(T, R):
    base, size = LY.scratch_layout(6, T, R)
    assert LY.scratch_layout(6, T, R, 1) == (base, size)
    for tp in (2, 4):
        regions, size_tp = LY.scratch_layout(6, T, R, tp)
        assert size_tp < size and regions.keys() == base.keys()
        sizes = _sizes(regions, size_tp)
        ref = _sizes(base, size)
        for name in ("qp", "q", "og", "ip", "mid", "part"):
            assert sizes[f"L0.{name}"] < ref[f"L0.{name}"], name
        for name in ("qi", "ilog", "sel", "h1", "h2", "dpart", "a.mixd", "logit"):
            assert sizes[f"L0.{name}"] == ref[f"L0.{name}"], name


def _sizes(regions, total):
    names = sorted(regions, key=regions.get)
    ends = [regions[n] for n in names[1:]] + [total]
    return {n: e - regions[n] for n, e in zip(names, ends)}


# ---------------------------------------------------------------- fake SGLang layers
class _Mod(torch.nn.Module):
    pass


def _param(module, path, t, **attrs):
    *parents, name = path.split(".")
    for p in parents:
        if not hasattr(module, p):
            setattr(module, p, _Mod())
        module = getattr(module, p)
    p = torch.nn.Parameter(t, requires_grad=False)
    for k, v in attrs.items():
        setattr(p, k, v)
    setattr(module, name, p)


def _empty(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device=META)


def fake_layer(kind: str, tp: int, ple: bool = False):
    """A decoder layer's params the kernel reads, at SGLang's per-rank shapes."""
    s = SGLANG[tp]
    layer = _Mod()
    f8, f32, u8 = torch.float8_e4m3fn, torch.float32, torch.uint8
    for path, (shape, _) in W._hc_tensors("attn_hyper_connection").items():
        _param(layer, path, _empty(*shape))
    for path, (shape, _) in W._hc_tensors("mlp_hyper_connection").items():
        _param(layer, path, _empty(*shape))
    _param(layer, "mlp.gate.weight", _empty(512, HIDDEN))
    _param(layer, "mlp.shared_expert_gate.weight", _empty(1, HIDDEN))
    ex = _Mod()
    layer.mlp.experts = ex
    ex.num_fused_shared_experts = 1
    _param(ex, "w13_weight", _empty(513, s["w13"], HIDDEN // 2, dtype=u8), is_shuffled=True)  # fmt: skip
    _param(ex, "w2_weight", _empty(513, HIDDEN, s["w2"], dtype=u8), is_shuffled=True)
    _param(ex, "w13_weight_scale", _empty(513, s["w13s"], HIDDEN // 32, dtype=u8))
    _param(ex, "w2_weight_scale", _empty(513, HIDDEN, s["w2s"], dtype=u8))

    def fp8(name, n, k):
        _param(layer, name + ".weight", _empty(n, k, dtype=f8))
        _param(layer, name + ".weight_scale", _empty(n, dtype=f32))

    if kind == "gdn":
        fp8("linear_attn.in_proj_qkvz", s["qkvz"], HIDDEN)
        fp8("linear_attn.in_proj_ba", s["ba"], HIDDEN)
        fp8("linear_attn.out_proj", HIDDEN, s["out_k"])
        _param(layer, "linear_attn.A_log", _empty(s["hv"], dtype=f32))
        _param(layer, "linear_attn.dt_bias", _empty(s["hv"]))
        _param(layer, "linear_attn.norm.weight", _empty(128))
        _param(layer, "linear_attn.conv1d.weight", _empty(s["conv"], 1, 4))
    else:
        fp8("qkv_proj", s["qkv"], HIDDEN)
        fp8("indexer.index_qk_proj", 640, HIDDEN)
        fp8("o_proj", HIDDEN, s["out_k"])
        for name, n in (("q_norm", 256), ("k_norm", 256), ("indexer.q_layernorm", 128),
                        ("indexer.k_layernorm", 128)):  # fmt: skip
            _param(layer, name + ".weight", _empty(n))
    if ple:
        layer.ple = _Mod()
        for name, shape in (("key_proj", (10240, HIDDEN)), ("value_proj", (HIDDEN, HIDDEN)),
                            ("norm_key", (10240,)), ("norm_query", (10240,)),
                            ("norm_conv", (10240,)), ("conv1d", (10240, 1, 4))):  # fmt: skip
            _param(layer, f"ple.{name}.weight", _empty(*shape))
    return layer


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("kind", ["gdn", "qsa"])
def test_layer_reasons_accept_the_rank_shards(kind, tp):
    assert W.layer_reasons(fake_layer(kind, tp, ple=kind == "gdn"), 1, kind, tp) == []


@pytest.mark.parametrize("kind", ["gdn", "qsa"])
@pytest.mark.parametrize("have,want", [(1, 2), (2, 1), (2, 4), (4, 2)])
def test_layer_reasons_refuse_another_tp(kind, have, want):
    reasons = W.layer_reasons(fake_layer(kind, have), 0, kind, want)
    assert reasons and all(r.startswith("layer 0: ") for r in reasons)
    # the MoE and the projections are both checked
    assert any("experts." in r for r in reasons)
    assert any("proj" in r for r in reasons)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_check_moe_params_at_tp(tp):
    s = SGLANG[tp]
    u8 = torch.uint8
    m = {
        "gu": _empty(513, s["w13"], HIDDEN // 2, dtype=u8),
        "gu_s": _empty(513, s["w13s"], HIDDEN // 32, dtype=u8),
        "dn": _empty(513, HIDDEN, s["w2"], dtype=u8),
        "dn_s": _empty(513, HIDDEN, s["w2s"], dtype=u8),
    }
    packing.check_moe_params(m, tp)
    for other in {1, 2, 4} - {tp}:
        with pytest.raises(AssertionError):
            packing.check_moe_params(m, other)


def _ngram_layer(rows, start):
    ple = _Mod()
    _param(ple, "ple_embedding.ngram_embedding.weight", _empty(rows, 160, dtype=torch.float8_e4m3fn))  # fmt: skip
    ple.ple_embedding.ngram_embedding.shard_indices = type(
        "S", (), {"org_vocab_start_index": start}
    )()
    return ple


@pytest.mark.parametrize("tp", [2, 4])
def test_ngram_reasons_check_the_vocab_shard(tp):
    rows = SGLANG[tp]["ngram"]
    assert W.ngram_reasons(_ngram_layer(rows, rows * (tp - 1)), tp) == []
    assert W.ngram_shard_start(_ngram_layer(rows, rows).ple_embedding.ngram_embedding) == rows  # fmt: skip
    # the TP 1 table, or a shard of another TP, would be indexed out of bounds
    assert W.ngram_reasons(_ngram_layer(SGLANG[1]["ngram"], 0), tp)
    assert W.ngram_reasons(_ngram_layer(rows // 2, 0), tp)


def test_proj_tables_at_tp():
    for tp in (1, 2, 4):
        g = LY.geom(tp)
        gdn, qsa = W.gdn_proj(tp), W.qsa_proj(tp)
        assert gdn["ip"][1] == g.QKVZ and gdn["out"][2] == g.VAL_DIM
        assert gdn["ba"][1] == g.BA_ROWS
        assert qsa["qp"][1] + qsa["qp2"][1] == g.QP_ROWS and qsa["qp2"][1] == 640
        assert qsa["out"][2] == g.VAL_DIM
    assert W.gdn_proj(1) == W.GDN_PROJ and W.qsa_proj(1) == W.QSA_PROJ


@pytest.mark.parametrize("tp", [2, 4])
def test_unshuffle_moe_expert_at_tp(tp):
    # a rank's chunk of the checkpoint's experts, as FusedMoE pads and AITER shuffles
    # it, comes back from the packing's inverse
    try:
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility.fp4_utils import e8m0_shuffle
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"AITER does not import: {type(e).__name__}: {e}")
    g = LY.geom(tp)
    E, I, IP = 2, g.MOE_INTER, g.MOE_INTER_PAD
    gen = torch.Generator().manual_seed(tp)

    def b(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)

    def scales(s):
        e, r, _ = s.shape
        return e8m0_shuffle(s.reshape(e * r, -1).contiguous()).view(e, r, -1)

    gate, up, down = b(E, I, HIDDEN // 2), b(E, I, HIDDEN // 2), b(E, HIDDEN, I // 2)
    gate_s, up_s, down_s = b(E, I, 80), b(E, I, 80), b(E, HIDDEN, I // 32)
    w13 = torch.zeros(E, 2 * IP, HIDDEN // 2, dtype=torch.uint8)
    w13[:, :I], w13[:, IP : IP + I] = gate, up
    s13 = torch.zeros(E, 2 * IP, 80, dtype=torch.uint8)
    s13[:, :I], s13[:, IP : IP + I] = gate_s, up_s
    w2 = torch.zeros(E, HIDDEN, IP // 2, dtype=torch.uint8)
    w2[..., : I // 2] = down
    s2 = torch.zeros(E, HIDDEN, g.DN_SGROUPS, dtype=torch.uint8)
    s2[..., : I // 32] = down_s
    m = {"gu": shuffle_weight(w13.contiguous(), (16, 16)), "gu_s": scales(s13),
         "dn": shuffle_weight(w2.contiguous(), (16, 16)), "dn_s": scales(s2)}  # fmt: skip
    for e in range(E):
        got = packing.unshuffle_moe_expert(m, e, tp)
        for x, r in zip(got, (gate[e], up[e], down[e], gate_s[e], up_s[e], down_s[e])):
            assert torch.equal(x, r)
