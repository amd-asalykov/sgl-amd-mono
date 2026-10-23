"""The Qwen3.8-Flash-Next gfx950 monokernel specs on CPU (RFC #43949): their registry
entries, the configurations they refuse before load, the checkpoint tensors the
load hook keeps, the launch plan of a verify step, and the draft steps the MTP
draft's switch takes. SGLang checks the declared requirements, the same way for
every spec."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry
try:
    from sglang.srt.kernel_plugins.amd.binding import covered_capture_sizes
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.guards import covered_capture_sizes

import sgl_amd_mono
from sgl_amd_mono.qwen38_flash_next import TEXT_CONFIG, draft, spec
from sgl_amd_mono.qwen38_flash_next import weights as W
from sgl_amd_mono.qwen38_flash_next.gfx950.runner import target_plans

LAYERS = 48
TEXT = SimpleNamespace(
    **TEXT_CONFIG,
    num_hidden_layers=LAYERS,
    layer_types=[
        "full_attention" if i % 4 == 3 else "linear_attention" for i in range(LAYERS)
    ],
    max_position_embeddings=262144,
    eos_token_id=248044,
)
VL = SimpleNamespace(
    text_config=TEXT, architectures=["Qwen4ExpForConditionalGeneration"]
)
# the NEXTN recipe's resolved server arguments the spec reads
CFG = SimpleNamespace(
    speculative_algorithm="EAGLE",
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=4,
    context_length=None,
)


def _replace(ns, **changes):
    return SimpleNamespace(**{**vars(ns), **changes})


def _config_reasons(cfg=CFG, hf_config=VL):
    with mock.patch.object(spec, "_has_flydsl", return_value=True):
        return spec._config_disable_reasons(cfg, hf_config)


def test_the_plugin_registers_the_target_and_its_draft():
    sgl_amd_mono.register_specs()
    for arch, want in (
        ("Qwen4ExpForConditionalGeneration", spec.SPEC),
        ("Qwen4ExpForCausalLMMTP", draft.SPEC),
    ):
        found, why = registry.find_spec(
            backend=sgl_amd_mono.BACKEND, architectures=[arch], isa="gfx950"
        )
        assert found is want, why


def test_the_recipe_is_accepted():
    assert _config_reasons() == []
    assert _config_reasons(cfg=_replace(CFG, speculative_num_draft_tokens=2)) == []


@pytest.mark.parametrize(
    "cfg, text",
    [
        (_replace(CFG, speculative_algorithm=None), TEXT),
        (_replace(CFG, speculative_num_draft_tokens=5), TEXT),
        (_replace(CFG, context_length=300000), TEXT),
        (CFG, _replace(TEXT, hidden_size=4096)),
        (CFG, _replace(TEXT, num_experts=256)),
        (CFG, _replace(TEXT, ple_layer_ids=[2, 6])),
        (CFG, _replace(TEXT, layer_types=["linear_attention"] * LAYERS)),
    ],
    ids=[
        "no speculation",
        "draft tokens",
        "context",
        "hidden size",
        "experts",
        "ple layers",
        "layer pattern",
    ],
)
def test_each_unsupported_setting_is_one_reason(cfg, text):
    assert len(_config_reasons(cfg=cfg, hf_config=_replace(VL, text_config=text))) == 1


def test_the_requirements_name_the_validated_setup():
    req = spec.SPEC.requires
    assert req.env == {
        "SGLANG_USE_AITER": True,
        "SGLANG_USE_AITER_FP8_PER_TOKEN": True,
        "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
    }
    assert req.server_args["speculative_eagle_topk"] == {1}
    assert req.attention_backends == {"aiter", "triton"}
    assert spec.SPEC.devices["gfx950"].tp_sizes == (1,)
    # every bound names <launch>.<output>.<metric> (Numerics checks it at import)
    assert {n.split(".")[0] for n in spec.NUMERICS.bounds} == {"verify"}
    assert {n.split(".")[0] for n in draft.NUMERICS.bounds} == {"draft"}


@pytest.mark.parametrize("tokens", [2, 3, 4])
def test_verify_batches_of_one_to_eight_are_captured(tokens):
    # a launch takes 32 rows; verify batches up to the configured 8 are aligned
    assert covered_capture_sizes(
        widths=spec.SPEC.widths("gfx950"), tokens_per_request=tokens, sizes=[1, 2, 4, 8]
    ) == list(range(1, 9))


def test_a_verify_step_is_thirteen_launches_over_every_layer():
    plans = target_plans(LAYERS)
    assert len(plans) == 13
    assert [i for _, ids in plans for i in ids] == list(range(LAYERS))
    assert plans[0][0] == ("gdn",) and plans[1][0] == ("gdn", "gdn", "qsa")
    assert all(kinds == ("gdn", "gdn", "gdn", "qsa") for kinds, _ in plans[2:])


def _t(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


def test_the_target_keeps_what_the_launches_pack_and_the_checks_read():
    copies: dict = {}
    model = SimpleNamespace(config=VL)
    p = "model.language_model."
    names = [
        f"{p}layers.0.attn_hyper_connection.hc_norm.weight",
        f"{p}layers.0.linear_attn.A_log",
        f"{p}layers.1.ple.key_proj.weight",
        f"{p}layers.3.self_attn.q_norm.weight",
        f"{p}layers.3.mlp.experts.255.down_proj.weight_scale",  # a checked expert
        f"{p}layers.3.mlp.shared_expert.up_proj.weight",
        f"{p}layers.0.linear_attn.in_proj_b.weight",  # a checked projection
        f"{p}hyper_connection_mixer.hc_norm.weight",
    ]
    skipped = [
        f"{p}layers.3.mlp.experts.7.down_proj.weight",  # not a checked expert
        f"{p}layers.4.linear_attn.in_proj_b.weight",  # not a checked layer
        f"{p}layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
        "mtp.layers.0.mlp.gate.weight",  # the draft's
        "lm_head.weight",
    ]
    for n in names + skipped:
        spec._keep_tensor(copies, model, n, _t(2))
    assert sorted(map(str, copies)) == sorted(
        map(
            str,
            [
                ("pack", 0, "attn_hyper_connection.hc_norm.weight"),
                ("pack", 0, "linear_attn.A_log"),
                ("pack", 1, "ple.key_proj.weight"),
                ("pack", 3, "self_attn.q_norm.weight"),
                ("check", 3, "mlp.experts.255.down_proj.weight_scale"),
                ("check", 3, "mlp.shared_expert.up_proj.weight"),
                ("check", 0, "linear_attn.in_proj_b.weight"),
                ("pack", "final", "hyper_connection_mixer.hc_norm.weight"),
            ],
        )
    )
    assert all(t.device.type == "cpu" for t in copies.values())
    # what a checkpoint without these tensors would leave unseen
    assert "layers.2.linear_attn.A_log" in W.target_unseen(copies, spec._kinds(VL))


def test_the_draft_keeps_the_mtp_head():
    copies: dict = {}
    for n in (
        "mtp.fc_hidden.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.experts.511.gate_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight_scale",
        "mtp.layers.0.mlp.experts.3.gate_proj.weight",
        "model.language_model.layers.0.mlp.gate.weight",
    ):
        draft._keep_tensor(copies, None, n, _t(2))
    assert set(copies) == {
        ("pack", "mtp", "fc_hidden.weight"),
        ("pack", "mtp", "layers.0.mlp.gate.weight"),
        ("check", "mtp", "mlp.experts.511.gate_proj.weight"),
        ("check", "mtp", "self_attn.o_proj.weight_scale"),
    }


def test_read_checkpoint_reads_only_the_tensors_kept(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {"mtp.fc_hidden.weight": _t(2), "lm_head.weight": _t(3)},
        str(tmp_path / "a.safetensors"),
    )
    save_file({"mtp.layers.0.mlp.gate.weight": _t(4)}, str(tmp_path / "b.safetensors"))
    read = []

    def loads(copies, name, tensor):
        def tracked():
            read.append(name)
            return tensor()

        draft._keep_tensor(copies, None, name, tracked)

    copies: dict = {}
    W.read_checkpoint(copies, str(tmp_path), loads)
    assert set(copies) == {
        ("pack", "mtp", "fc_hidden.weight"),
        ("pack", "mtp", "layers.0.mlp.gate.weight"),
    }
    assert all(t.device.type == "cpu" for t in copies.values())
    assert sorted(read) == ["mtp.fc_hidden.weight", "mtp.layers.0.mlp.gate.weight"]


def test_both_specs_read_their_copies_from_the_checkpoint(tmp_path):
    from safetensors.torch import save_file

    p = "model.language_model."
    save_file(
        {
            f"{p}layers.0.linear_attn.A_log": _t(2),
            f"{p}layers.4.mlp.experts.7.down_proj.weight": _t(3),
            "mtp.fc_hidden.weight": _t(2),
            "lm_head.weight": _t(3),
        },
        str(tmp_path / "model.safetensors"),
    )
    target_copies: dict = {}
    spec._read_copies(target_copies, SimpleNamespace(config=VL), str(tmp_path))
    assert set(target_copies) == {("pack", 0, "linear_attn.A_log")}
    draft_copies: dict = {}
    draft._read_copies(draft_copies, None, str(tmp_path))
    assert set(draft_copies) == {("pack", "mtp", "fc_hidden.weight")}


class _Mode:
    def __init__(self, name):
        self.name = name

    def is_draft_extend_v2(self):
        return self.name == "DRAFT_EXTEND_V2"

    def is_decode(self):
        return self.name == "DECODE"


def _draft_step(mode, R, rows, dtype=torch.bfloat16, width=draft.HC_DIM):
    hs = torch.zeros(rows, width, dtype=dtype)
    fb = SimpleNamespace(
        forward_mode=_Mode(mode),
        batch_size=R,
        spec_info=SimpleNamespace(hidden_states=hs),
    )
    return fb, torch.zeros(rows, dtype=torch.int64)


@pytest.mark.parametrize(
    "mode, R, rows, taken",
    [
        ("DRAFT_EXTEND_V2", 3, 12, True),
        ("DECODE", 8, 8, True),
        ("DRAFT_EXTEND_V2", 3, 9, False),  # not R x num_draft_tokens
        ("DECODE", 9, 9, False),  # past the 8 requests a launch takes
        ("TARGET_VERIFY", 2, 8, False),
        ("EXTEND", 1, 40, False),
    ],
)
def test_the_draft_switch_takes_draft_extend_and_decode(mode, R, rows, taken):
    d = draft._Draft.__new__(draft._Draft)
    d.runner = SimpleNamespace(max_reqs=8, L_ext=4)
    fb, ids = _draft_step(mode, R, rows)
    assert bool(d.rows_of(fb, ids, None)) == taken


def test_the_draft_switch_refuses_other_streams():
    d = draft._Draft.__new__(draft._Draft)
    d.runner = SimpleNamespace(max_reqs=8, L_ext=4)
    fb, ids = _draft_step("DECODE", 2, 2, dtype=torch.float32)
    assert not d.rows_of(fb, ids, None)
    fb, ids = _draft_step("DECODE", 2, 2)
    assert not d.rows_of(fb, ids, torch.zeros(2, 2560))  # input embeds


def test_the_draft_switch_runs_before_the_runtime_switch():
    calls = []

    class Model:
        def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kw):
            calls.append("standard")

    model = Model()
    d = SimpleNamespace(
        rows_of=lambda fb, ids, emb: fb == "draft",
        step=lambda ids, pos, fb: calls.append("launch"),
    )
    draft._route_draft_steps(model, d)
    model.forward(None, None, "draft")
    model.forward(None, None, "extend")
    assert calls == ["launch", "standard"]


def test_more_requests_than_a_launch_takes_run_sglangs_forward():
    calls = []

    class LM:
        def forward(
            self,
            input_ids,
            positions,
            forward_batch,
            get_embedding=False,
            pp_proxy_tensors=None,
        ):
            calls.append(forward_batch)
            return "standard"

    built = spec._Built.__new__(spec._Built)
    built.lm, built.runner, built.check = LM(), SimpleNamespace(max_reqs=8), False
    key = registry.ShapeKey(
        forward_mode="target_verify", batch_size=9, tokens_per_request=2
    )
    assert built.forward(registry.Plan(key=key), None, None, "fb") == "standard"
    assert calls == ["fb"]


def test_check_compares_every_region_and_reports_violations(caplog):
    import logging

    Numerics, NumericsKind = registry.Numerics, registry.NumericsKind

    from sgl_amd_mono.qwen38_flash_next import check as C

    kv = torch.zeros(8, 4)
    ring = torch.zeros(3, 4)
    ngram = torch.zeros(2, 2, dtype=torch.int64)
    loc = torch.tensor([1, 5])
    regions = {
        "kv.3.k": (kv, loc),
        "kv.7.k": (kv, torch.tensor([2])),
        "ring.3": (ring, None),
        "ngram": (ngram, slice(0, 1)),
    }
    pre = C.snapshot(regions)
    # SGLang's forward: rows 1 and 5 of the KV, ring row 0, one n-gram context
    kv[loc] = torch.tensor([[1.0, 2, 3, 4], [4.0, 3, 2, 1]])
    ring[0] = 1.0
    ngram[0] = torch.tensor([7, 9])
    ref = C.snapshot(regions)
    C.restore(regions, pre)
    assert not kv.any() and not ring.any() and not ngram.any()
    # the launches: a small error on one KV row, the ring and n-gram exact
    kv[loc] = torch.tensor([[1.0, 2, 3, 4.1], [4.0, 3, 2, 1]])
    ring[0] = 1.0
    ngram[0] = torch.tensor([7, 9])
    logits = torch.tensor([[1.0, 2.0]])
    values = C.compare(
        "verify", regions, ref, pre, {"logits": (logits, logits.clone())}
    )
    assert values["verify.kv.rel"] == pytest.approx(
        0.1 / 60**0.5, rel=1e-5
    )  # L2 over both rows
    assert values["verify.kv.rows"] == 2  # the unchanged row of layer 7 does not count
    assert values["verify.ring.rel"] == 0 and values["verify.ring.rows"] == 1
    assert values["verify.ngram.max_abs"] == 0
    assert values["verify.logits.cosine"] == pytest.approx(1.0)
    bounds = Numerics(
        NumericsKind.TOLERANCE, {"verify.kv.rel": 0.01, "verify.ngram.max_abs": 0}
    )
    with caplog.at_level(logging.INFO, logger=C.__name__):
        C.report("verify 2 x 1", values, bounds)
    violations = [
        r.getMessage() for r in caplog.records if "violation" in r.getMessage()
    ]
    assert violations == [
        "monokernel check violation: verify 2 x 1 verify.kv.rel = 0.0129099, bound 0.01"
    ]


def test_the_launches_take_the_rope_positions_sglang_applies():
    # with MRoPE on (the VL model), SGLang's forward rotates by mrope_positions, which
    # differ from the sequence positions in a request with images
    fb = SimpleNamespace(
        batch_size=1,
        input_ids=torch.zeros(4, dtype=torch.int64),
        positions=torch.arange(122, 126),
        mrope_positions=torch.arange(66, 70).expand(3, -1),
        out_cache_loc=torch.arange(4),
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        seq_lens=torch.tensor([122]),
    )
    built = spec._Built.__new__(spec._Built)
    for mrope, want in ((True, fb.mrope_positions), (False, fb.positions)):
        built.lm = SimpleNamespace(is_mrope_enabled=mrope)
        assert torch.equal(built._batch(fb).positions, want)
