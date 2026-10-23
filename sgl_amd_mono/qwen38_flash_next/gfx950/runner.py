# SPDX-License-Identifier: Apache-2.0
"""Host side of the Qwen3.8-Flash-Next launches: the pointer tables, the launch
sequence of a target verify step (13 launches) and of an MTP draft step (one).

A runner holds the packed weights (``packing.py``), the scratch mailboxes and the
guard block (``common/guard.py``), whose word 0 is the step epoch the mailbox tags
derive from. The pools and the step's batch tensors come with each call: the
runner reads SGLang's pools through the accessors their classes provide (the
spec names them in ``SGLANG_READS``), so a test can hand it pools of its own.
Pointer tables are cached by the addresses they hold; a captured table's H2D copy
re-reads its pinned source at every replay, so the source is kept alive.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import msgspec
import torch

from . import layout as LY
from . import packing
from .config import HC_DIM, HIDDEN

MAX_REQS = LY.MAX_R
# eager steps key their tables by fresh addresses; past this many the cache restarts
_TABLE_CACHE = 512


class StepBatch(msgspec.Struct, frozen=True):
    """The batch tensors a launch reads, as the ForwardBatch holds them (int64)."""

    batch_size: int
    input_ids: torch.Tensor
    positions: torch.Tensor  # [rows] or [3, rows] (MRoPE axes)
    out_cache_loc: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor

    @classmethod
    def of(cls, forward_batch) -> StepBatch:
        fb = forward_batch
        return cls(
            batch_size=int(fb.batch_size),
            input_ids=fb.input_ids,
            positions=fb.positions,
            out_cache_loc=fb.out_cache_loc,
            req_pool_indices=fb.req_pool_indices,
            seq_lens=fb.seq_lens,
        )

    def ptrs(self) -> tuple:
        return tuple(
            t.data_ptr()
            for t in (
                self.input_ids,
                self.positions,
                self.out_cache_loc,
                self.req_pool_indices,
                self.seq_lens,
            )
        )

    def check(self) -> None:
        for name in ("input_ids", "positions", "out_cache_loc", "req_pool_indices"):
            t = getattr(self, name)
            if t.dtype != torch.int64:
                raise ValueError(f"monokernel: {name} is {t.dtype}, not int64")


def ensure_writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache when the configured one is read-only."""
    from pathlib import Path

    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def _default_build():
    from .kernel import build_qwen38_launch

    return build_qwen38_launch


class _Launches:
    """What the target and draft runners share: the scratch mailboxes, the guard
    block and the launcher cache."""

    def __init__(self, scratch_bytes: int, device, build: Optional[Callable]):
        from sgl_amd_mono.common import guard

        ensure_writable_flydsl_cache()
        self.device = device
        self.scratch = torch.zeros(scratch_bytes // 4, dtype=torch.int32, device=device)
        # word 0: the step epoch (mailbox tags), then the deadline and error words
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        self.build = build or _default_build()
        # SGLANG_AMD_MONO_FAULT_INJECT: the wait one launch kind stalls at (tests)
        self.fault = guard.fault_injection(0) if build is None else None
        self._launchers: dict = {}
        self._tables: dict = {}
        self._pinned: list = []

    def set_deadline(self, seconds: float) -> None:
        from sgl_amd_mono.common import guard

        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def error_words(self) -> list[int]:
        from sgl_amd_mono.common import guard

        return [int(self.epoch[i]) for i in guard.ERROR_WORDS]

    def _launcher(self, *args, **kwargs):
        if self.fault is not None:
            kwargs.update(fault_site=self.fault[0], fault_after=self.fault[1])
        key = (args, tuple(sorted(kwargs.items())))
        if key not in self._launchers:
            self._launchers[key] = self.build(*args, **kwargs)
        return self._launchers[key]

    def _device_table(self, rows: list) -> torch.Tensor:
        host = torch.tensor(rows, dtype=torch.int64)
        if torch.cuda.is_current_stream_capturing():
            pinned = host.pin_memory()
            self._pinned.append(pinned)
            return pinned.to(self.device, non_blocking=True)
        return host.to(self.device)

    def _remember(self, key, value):
        if len(self._tables) >= _TABLE_CACHE and not (
            torch.cuda.is_current_stream_capturing()
        ):
            self._tables = {k: v for k, v in self._tables.items() if k[0] == "captured"}
        tag = "captured" if torch.cuda.is_current_stream_capturing() else "eager"
        self._tables[(tag, *key)] = value
        return value

    def _cached(self, key):
        return self._tables.get(("captured", *key)) or self._tables.get(("eager", *key))


class TargetWeights(msgspec.Struct, frozen=True):
    """What the target launches read besides SGLang's pools: each layer's packed
    tensors (``packing.pack_gdn_layer`` / ``pack_qsa_layer``), layer 1's PLE, the
    final mixer, the token embedding and the PLE n-gram table."""

    kinds: tuple[str, ...]  # "gdn" | "qsa" per layer
    layers: dict  # layer id -> packed dict
    ple: dict  # packing.pack_ple
    final: dict  # {"norm", "down", "up"}
    embed_w: torch.Tensor
    ngram_table: torch.Tensor  # FP8 [rows, NGRAM_DIM]
    ngram_scale: torch.Tensor
    ngram_meta: torch.Tensor  # int64 [3 + 2 * NGRAM_HEADS]
    eos_token_id: int


def target_plans(n_layers: int) -> list[tuple[tuple[str, ...], list[int]]]:
    """The launches of a verify step: layer 0 (with the embedding and the PLE
    n-gram prelude), layers 1-3 (PLE first), then 3 GDN + 1 QSA layers each."""
    plans = [(("gdn",), [0]), (("gdn", "gdn", "qsa"), [1, 2, 3])]
    for b in range(1, n_layers // 4):
        plans.append((("gdn", "gdn", "gdn", "qsa"), list(range(4 * b, 4 * b + 4))))
    return plans


PLE_LAUNCH = 1


class TargetRunner(_Launches):
    """Target verify of R <= 8 requests x L = num_draft_tokens rows (topk 1): the
    whole model (embedding, PLE, every decoder layer, final mixer) in 13 launches."""

    def __init__(self, weights: TargetWeights, L: int, device, build=None):
        if not 1 <= L <= 4:
            raise ValueError(f"the launches take 1..4 tokens a request, not {L}")
        self.w, self.L = weights, L
        self.max_reqs = min(MAX_REQS, LY.MAX_T // L)
        self.plans = target_plans(len(weights.kinds))
        # one scratch for every R: a launch accepts only pairs tagged with its own epoch
        nbytes = max(
            LY.scratch_layout(4 + 2, r * L, r)[1] for r in range(1, self.max_reqs + 1)
        )
        super().__init__(nbytes, device, build)
        S = self.max_reqs * L
        self.h_bufs = [
            torch.empty(S, HC_DIM, dtype=torch.bfloat16, device=device)
            for _ in range(2)
        ]
        self.ssq_bufs = [
            torch.zeros(S, 4, LY.NPART, dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        self.emb_buf = torch.empty(S, HIDDEN, dtype=torch.bfloat16, device=device)
        self.x_buf = torch.empty(S, HIDDEN, dtype=torch.bfloat16, device=device)

    def forward(self, batch: StepBatch, rtp, kvp, cos: torch.Tensor):
        """One verify step -> (final mixer output x [S, HIDDEN], residual streams
        after the last layer [S, HC_DIM])."""
        R = batch.batch_size
        S = R * self.L
        tables, meta = self._tables_of(batch, rtp, kvp, cos)
        stream = torch.cuda.current_stream()
        h = self.h_bufs[1]  # unread: layer 0 reads the embedding rows
        for i in range(len(self.plans)):
            out = self.h_bufs[i % 2]
            args, kwargs = self.launch_key(i, R, meta)
            self._launcher(*args, **kwargs)(
                h.data_ptr(),
                out.data_ptr(),
                tables[i].data_ptr(),
                self.scratch.data_ptr(),
                self.epoch.data_ptr(),
                i,
                stream=stream,
            )
            h = out
        self.epoch[:1].add_(1)
        return self.x_buf[:S], h[:S]

    def launch_key(self, i: int, R: int, meta: dict) -> tuple[tuple, dict]:
        """The build arguments of launch ``i`` of a step of R requests."""
        args = (
            self.plans[i][0],
            R,
            self.L,
            meta["rtt_stride"],
            meta["pos_stride"],
            meta["map_i64"],
        )
        return (
            args,
            dict(
                ple_first=i == PLE_LAUNCH,
                final_mix=i == len(self.plans) - 1,
                prelude=i == 0,
                eos_token_id=self.w.eos_token_id,
                iconv_strides=meta["iconv"],
                issm_row=meta["issm_row"],
                ple_ist_row=meta["ple_ist_row"],
                ictx_row=meta["ictx_row"],
                first_src="embed" if i == 0 else "global",
                ssm_bf16=meta["ssm_bf16"],
                seq_p0=True,  # sequence positions from seq_lens: the positions may be MRoPE's
            ),
        )

    def launch_keys(self, batch: StepBatch, rtp, kvp, cos) -> list:
        """Every launch's build arguments for this step's shape and pools."""
        _, meta = self._tables_of(batch, rtp, kvp, cos)
        return [
            self.launch_key(i, batch.batch_size, meta) for i in range(len(self.plans))
        ]

    def _tables_of(self, batch: StepBatch, rtp, kvp, cos: torch.Tensor):
        key = (batch.batch_size, *batch.ptrs(), cos.data_ptr())
        hit = self._cached(key)
        if hit is not None:
            return hit
        batch.check()
        return self._remember(key, self._build_tables(batch, rtp, kvp, cos))

    def _ngram(self, rtp, R: int):
        ctx, ictx = rtp.ngram_pool.context, rtp.ngram_pool.intermediate_context
        if ctx.dtype != torch.int64 or not ctx.is_contiguous() or ctx.shape[1] != 2:
            raise ValueError("monokernel: the n-gram context must be int64 [slots, 2]")
        if (
            ictx is None
            or ictx.dtype != torch.int64
            or ictx.shape[1] < self.L
            or ictx.shape[0] < R
            or ictx.stride(1) != 2
            or ictx.stride(2) != 1
        ):
            raise ValueError("monokernel: the n-gram verify context's layout")
        return ctx, ictx

    def _global_row(self, i, batch, rtp, kvp, cos, ngram):
        ctx, ictx = ngram
        w = self.w
        g = packing.global_ptr_row(
            batch.positions,
            batch.out_cache_loc,
            batch.req_pool_indices,
            batch.seq_lens,
            rtp.req_to_token,
            cos,
            kvp.qsa_rope_position_buffer,
            rtp.req_index_to_mamba_index_mapping,
        )
        g[LY.G_PLE_EMB] = self.emb_buf.data_ptr()
        g[LY.G_FIN_OUT] = self.x_buf.data_ptr()
        g[LY.G_INPUT_IDS] = batch.input_ids.data_ptr()
        g[LY.G_EMBED_W] = w.embed_w.data_ptr()
        g[LY.G_NGRAM_W] = w.ngram_table.data_ptr()
        g[LY.G_NGRAM_SCALE] = w.ngram_scale.data_ptr()
        g[LY.G_NGRAM_CTX] = ctx.data_ptr()
        g[LY.G_NGRAM_META] = w.ngram_meta.data_ptr()
        g[LY.G_NGRAM_ICTX] = ictx[0].data_ptr()
        # the residual stream of launch i: h_bufs[(i - 1) % 2] -> h_bufs[i % 2]
        g[LY.G_HIN_SSQ] = self.ssq_bufs[(i - 1) % 2].data_ptr()
        g[LY.G_HOUT_SSQ] = self.ssq_bufs[i % 2].data_ptr()
        return g

    def _gdn_row(self, layer_id, t, R, rtp, meta):
        cache = rtp.mamba2_layer_cache(layer_id)
        conv, ssm = cache.conv[0], cache.temporal
        if not (conv.is_contiguous() and ssm.is_contiguous()):
            raise ValueError("monokernel: GDN states must be contiguous")
        if conv.dtype != torch.bfloat16 or ssm.dtype not in (
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError(f"monokernel: GDN state dtypes {conv.dtype} / {ssm.dtype}")
        if (
            conv[0].numel() * 2 != LY.CONV_SLOT_BYTES
            or ssm[0].numel() * 4 != LY.SSM_SLOT_BYTES
        ):
            raise ValueError("monokernel: GDN state slot sizes")
        meta.setdefault("ssm_bf16", ssm.dtype == torch.bfloat16)
        if meta["ssm_bf16"] != (ssm.dtype == torch.bfloat16):
            raise ValueError("monokernel: GDN layers differ in SSM state dtype")
        row = packing.gdn_ptr_row(t, conv, ssm)
        issm, icw = cache.intermediate_ssm, cache.intermediate_conv_window[0]
        if (
            issm is None
            or issm.dtype != ssm.dtype
            or not issm[0].is_contiguous()
            or issm.shape[1] < self.L
            or issm.shape[0] < R
            or icw.dtype != torch.bfloat16
            or icw.shape[0] < R
            or icw.shape[1] < self.L
        ):
            raise ValueError("monokernel: GDN verify scratch layout")
        # request r's verify scratch is row r (verify_intermediate_state_indices)
        row[LY.P_ISSM] = issm.data_ptr()
        row[LY.P_ICONV] = icw.data_ptr()
        for name, value in (
            ("iconv", (icw.stride(1), icw.stride(2), icw.stride(3), icw.stride(0))),
            ("issm_row", issm.stride(0)),
        ):
            if meta.setdefault(name, value) != value:
                raise ValueError(f"monokernel: GDN layers differ in {name}")
        return row

    def _ple_ptrs(self, row, layer_id, R, rtp, meta):
        packing.add_ple_ptrs(row, self.w.ple, rtp.short_conv_layer_cache(layer_id))
        istate = rtp.short_conv_layer_intermediate_cache(layer_id)
        if (
            istate is None
            or istate.dtype != torch.bfloat16
            or not istate[0].is_contiguous()
            or istate.shape[1] < self.L
            or istate.shape[0] < R
            or istate.stride(1) != HC_DIM * LY.PLE_STATE_LEN
        ):
            raise ValueError("monokernel: PLE verify scratch layout")
        row[LY.P_PLE_ISTATE] = istate.data_ptr()
        meta["ple_ist_row"] = istate.stride(0)

    def _qsa_row(self, layer_id, t, kvp):
        return packing.qsa_ptr_row(
            t,
            kvp.get_key_buffer(layer_id),
            kvp.get_value_buffer(layer_id),
            kvp.get_qsa_key_state_buffer(layer_id),
            kvp.get_qsa_compressed_k_buffer(layer_id),
        )

    def _final_row(self):
        f = self.w.final
        row = [0] * LY.N_PTR
        row[LY.P_HCA_NORM] = f["norm"].data_ptr()
        row[LY.P_HCA_DOWN] = f["down"].data_ptr()
        row[LY.P_HCA_UP] = f["up"].data_ptr()
        # the combine gate is unused (no combine): any layer's inject weight
        last = len(self.w.kinds) - 1
        row[LY.P_HCA_INJ] = self.w.layers[last]["hca"]["inj"].data_ptr()
        return row

    def _build_tables(self, batch, rtp, kvp, cos):
        R = batch.batch_size
        ngram = self._ngram(rtp, R)
        meta = {"ictx_row": ngram[1].stride(0)}
        tables = []
        for i, (plan, layers) in enumerate(self.plans):
            rows = []
            for kind, layer_id in zip(plan, layers):
                t = self.w.layers[layer_id]
                if kind == "gdn":
                    row = self._gdn_row(layer_id, t, R, rtp, meta)
                    if i == PLE_LAUNCH and layer_id == layers[0]:
                        self._ple_ptrs(row, layer_id, R, rtp, meta)
                else:
                    row = self._qsa_row(layer_id, t, kvp)
                rows.append(row)
            rows.append(self._global_row(i, batch, rtp, kvp, cos, ngram))
            if i == len(self.plans) - 1:
                rows.append(self._final_row())
            tables.append(self._device_table(rows))
        pos = batch.positions
        mamba_map = rtp.req_index_to_mamba_index_mapping
        meta.update(
            rtt_stride=rtp.req_to_token.stride(0),
            # 1-D text positions: the three MRoPE axes read the same element
            pos_stride=pos.stride(0) if pos.dim() == 2 else 0,
            map_i64=mamba_map.dtype == torch.int64,
        )
        return tables, meta


class DraftWeights(msgspec.Struct, frozen=True):
    """The MTP head as ``packing.pack_mtp`` packs it, its QSA layer's id in the
    draft's KV pool, the token embedding and the end-of-sequence id."""

    packed: dict
    layer_id: int
    embed_w: torch.Tensor
    eos_token_id: int


class DraftRunner(_Launches):
    """MTP (NEXTN) draft steps of R <= 8 requests in one launch each: the input
    fusion, the QSA layer and the draft mixer; draft extend runs R x L rows, draft
    decode R x 1."""

    def __init__(self, weights: DraftWeights, L_extend: int, device, build=None):
        if not 1 <= L_extend <= 4:
            raise ValueError(f"draft extend of 1..4 tokens a request, not {L_extend}")
        self.w, self.L_ext = weights, L_extend
        self.max_reqs = min(MAX_REQS, LY.MAX_T // L_extend)
        nbytes = max(
            LY.scratch_layout(1 + 2, r * t, r)[1]
            for r in range(1, self.max_reqs + 1)
            for t in (1, L_extend)
        )
        super().__init__(nbytes, device, build)
        S = self.max_reqs * L_extend
        self.x_buf = torch.empty(S, HIDDEN, dtype=torch.bfloat16, device=device)
        # draft decode step i + 1 reads step i's output: two buffers, alternating
        self.hc_bufs = [
            torch.empty(S, HC_DIM, dtype=torch.bfloat16, device=device)
            for _ in range(2)
        ]
        self._calls = 0
        self.ssq_buf = torch.zeros(S, 4, LY.NPART, dtype=torch.float32, device=device)
        # a QSA-only launch reads no mamba slot
        self.mamba_map = torch.zeros(4096, dtype=torch.int32, device=device)

    def forward(self, batch: StepBatch, hidden_states, kvp, req_to_token, cos):
        """One draft step -> (draft mixer output x [S, HIDDEN], residual streams
        [S, HC_DIM]); ``hidden_states`` [S, HC_DIM] is the target's (or the
        previous draft step's) residual stream."""
        R = batch.batch_size
        S = batch.input_ids.numel()
        table, meta = self._table_of(batch, hidden_states, kvp, req_to_token, cos)
        hc = self.hc_bufs[self._calls % 2]
        if hc.data_ptr() == hidden_states.data_ptr():
            self._calls += 1
            hc = self.hc_bufs[self._calls % 2]
        self._calls += 1
        args, kwargs = self.launch_key(R, S // R, meta)
        self._launcher(*args, **kwargs)(
            0,
            hc.data_ptr(),
            table.data_ptr(),
            self.scratch.data_ptr(),
            self.epoch.data_ptr(),
            0,
            stream=torch.cuda.current_stream(),
        )
        self.epoch[:1].add_(1)
        return self.x_buf[:S], hc[:S]

    def launch_key(self, R: int, L: int, meta: dict) -> tuple[tuple, dict]:
        args = (("qsa",), R, L, meta["rtt_stride"], meta["pos_stride"], False)
        return args, dict(
            final_mix=True, first_src="mtp", eos_token_id=self.w.eos_token_id
        )

    def _table_of(self, batch, hidden_states, kvp, req_to_token, cos):
        S = batch.input_ids.numel()
        key = (
            batch.batch_size,
            S,
            *batch.ptrs(),
            hidden_states.data_ptr(),
            cos.data_ptr(),
        )
        hit = self._cached(key)
        if hit is not None:
            return hit
        batch.check()
        lid = self.w.layer_id
        g = packing.global_ptr_row(
            batch.positions,
            batch.out_cache_loc,
            batch.req_pool_indices,
            batch.seq_lens,
            req_to_token,
            cos,
            kvp.qsa_rope_position_buffer,
            self.mamba_map,
        )
        g[LY.G_INPUT_IDS] = batch.input_ids.data_ptr()
        g[LY.G_EMBED_W] = self.w.embed_w.data_ptr()
        g[LY.G_MTP_HIN] = hidden_states.data_ptr()
        g[LY.G_FIN_OUT] = self.x_buf.data_ptr()
        g[LY.G_HOUT_SSQ] = self.ssq_buf.data_ptr()
        rows = packing.mtp_rows(
            self.w.packed,
            kvp.get_key_buffer(lid),
            kvp.get_value_buffer(lid),
            kvp.get_qsa_key_state_buffer(lid),
            kvp.get_qsa_compressed_k_buffer(lid),
            g,
        )
        pos = batch.positions
        meta = {
            "rtt_stride": req_to_token.stride(0),
            "pos_stride": pos.stride(0) if pos.dim() == 2 else 0,
        }
        return self._remember(key, (self._device_table(rows), meta))
