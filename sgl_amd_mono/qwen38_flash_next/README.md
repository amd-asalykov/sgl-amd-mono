# Qwen3.8-Flash-Next monokernel on gfx950

This document explains the monokernel in `sgl_amd_mono/qwen38_flash_next/` including what the target verify and MTP draft kernels do, stage by stage. File and symbol names refer to the plugin repo (sgl-amd-mono); the SGLang interface is the one of RFC sgl-project/sglang#43949.

## 1. Scope

| | target verify (`spec.py`) | MTP draft (`draft.py`) |
|---|---|---|
| model class | `Qwen4ExpForConditionalGeneration` | `Qwen4ExpForCausalLMMTP` |
| SGLang step | NEXTN `TARGET_VERIFY`, topk 1 | `DRAFT_EXTEND_V2` and the draft `DECODE` steps |
| rows | R <= 8 requests x L = k + 1 <= 4 tokens (<= 32 rows) | extend: R x L rows; decode: R rows |
| launches per step | 13 | 1 |
| larger batches | SGLang's forward | SGLang's forward |

GPU: gfx950 (MI355X), TP-1 only support. Weights: the Quark checkpoint `amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8`
(MXFP4 experts, W8A8 FP8 attention / GDN projections, FP8 PLE embedding).


## 2. The target verify step

### 2.1 Launch plan

`runner.target_plans` splits the 48 layers into 13 launches:

| launch | layers | also runs |
|---|---|---|
| 0 | 0 (GDN) | embedding lookup and the PLE n-gram prelude |
| 1 | 1-3 (GDN, GDN, QSA) | the PLE block before layer 1 |
| 2-12 | 3 GDN + 1 QSA each (layers 4-47) | launch 12 ends with the model-level HC mixer |

SGLang's logits processor then turns the final hidden states into logits, as on the normal path.

Why 4 layers per launch rather than one launch for the whole model: every layer is unrolled at compile time, so
the code size and the FlyDSL compile time grow with the layers per launch (a full-width build already takes several
minutes per batch size). Registers are not the limit. A launch boundary costs a few microseconds on a verify
step of about 4.4 ms (1 request) to 16.5 ms (8 requests), so 13 launches are close to the ideal.

Every launch is one grid of **256 CTAs x 512 threads** (8 waves), all resident for the whole launch.

### 2.2 Stages of one layer

Each row is (request r, draft position j) = row t = r * L + j with L=k+1=num_draft_tokens. All stages handle all rows of the step: the rows are the N columns of the MFMA tiles (ceil(rows / 16) column tiles sharing the weight registers), so one pass over a weight serves the whole batch.

**Attention-side HC mixer** (`stage_mix`, bf16 MFMA): RMS norm of the 4 branches, low-rank down / up projection,
sigmoid gates, the gated mean of the branches -> block input `x[t]` (2560), plus the combine-gate partials.

**GDN layer** (linear attention):

1. `stage_in_proj`: W8A8 FP8 GEMM of `x` to q, k, v, z (16,384 rows) plus the b / a gates; activations quantized
   per token (AITER's dynamic FP8 quantization).
2. `stage_core`: one task per (value head, V-row split), for all requests: the causal conv1d over each request's L
   tokens (from the committed conv window), l2-norm, and the gated delta rule token by token on the request's state
   held in registers. After every token it writes the state and the conv window into SGLang's verify scratch
   (`intermediate_ssm`, `intermediate_conv_window`, row r, step j), which SGLang's accept step commits from. Output
   `og = o * w * sigmoid(z)` with its sum-of-squares and max-|og| partials.
3. `stage_out_proj`: gated RMS norm, per-row FP8 quantization (scale from the partials), W8A8 `out_proj`, then the
   HC combine into the 4 residual branches.

**QSA layer** (full attention, sparse):

1. `stage_qsa_proj`: W8A8 GEMM of `x` to q (24 heads x 256), the new K / V (2 KV heads) and the indexer's q / k.
2. `stage_qsa_prep`: q / k norms and partial RoPE (64 of 256 dims, MRoPE positions when SGLang uses them); writes the
   new tokens' K / V into SGLang's KV cache at `out_cache_loc`, the indexer keys into the request's ring and, when a
   4-token group completes, its compressed key.
3. `stage_qsa_scores`: index logits of every complete 4-token block of the context for every row (MFMA over
   compressed keys, ReLU-sum over the 4 indexer heads).
4. `stage_qsa_select` / `stage_qsa_merge_select`: exact top-512 blocks per row: a radix select per 8192-block
   segment, then a merge of the segment winners for long contexts (more than one segment).
5. `stage_qsa_attn`: split flash-decoding over the selected tokens: one task per (row, chunk of 64 tokens) producing
   partial (o, max, sum) for all 24 heads. A row attends causally inside its own request: verify tokens of the same
   request at earlier positions are read from the stage-2 output.
6. `stage_qsa_merge`: combine the chunk partials, sigmoid output gate.
7. `stage_out_proj`: per-row FP8 quantization, W8A8 `o_proj`, HC combine.

**MoE block:**

1. `stage_mix` (MoE-side HC mixer), as above.
2. `stage_router`: bf16 router GEMM -> logits for 512 experts per row.
3. Routing (in every CTA, redundantly, so no hand-off is needed): per-row top-10 with renormalized softmax, plus the
   shared expert's gate; the *union* of the experts chosen by any row, sorted, becomes the step's expert *slots*.
4. `stage_gate_up`: per (slot, 16-row tile) task, the MXFP4 gate and up projections for all rows, read in place from
   SGLang's FusedMoE weights (AITER layout). Only routed (row, expert) pairs are published: `silu(g) * u`.
5. `stage_down`: experts are grouped (3-4 slots per group); each group stages its routed rows and runs the MXFP4 down
   projection, producing f32 partials. A second set of CTAs sums the groups' partials per output row with the routing
   weights and applies the HC combine.

### 2.3 Before and after the layers

- **Prelude** (launch 0): token embedding lookup; for each row, the 16 PLE n-gram ids from the request's token
  history and the step's tokens, the FP8 PLE embedding rows times their scales, and the n-gram context after each
  draft step (written to SGLang's verify scratch for the accept step).
- **PLE block** (launch 1): the bf16 key / value projections of the n-gram embedding, the branch gates, the group
  norm and the dilated short convolution over the tokens (its window after each token goes to verify scratch).
- **Final mixer** (launch 12): the model-level HC mixer that produces the hidden state the logits processor reads.

### 2.4 What the step writes

Exactly what SGLang's own verify step writes, for every row, so the accept / commit logic, the draft and later steps
see no difference:

- the KV cache at `out_cache_loc`, the QSA index-key ring, the compressed keys and their RoPE positions;
- the GDN, PLE and n-gram verify scratch (state after every draft position), which SGLang's accept step commits;
- the final hidden states for the logits processor, and the residual streams the MTP draft reads.

The committed GDN states, the KV of other requests and every other pool entry are untouched. The kernel test checks
this per step ("every step writes only its own rows").

## 3. Hand-offs between stages: tagged mailboxes

There is no kernel boundary between stages, so a consumer must know when the value it needs is written. The
launches use **tagged-pair mailboxes** (`common/sync.py`):

- Every value a stage hands on is stored in a global scratch buffer as an 8-byte pair: the 32-bit value (or a pair
  of bf16) and a 32-bit **tag**. The pair is written in a single store, so a reader never sees a new tag with an old
  value.
- The tag identifies the step and the launch: `tag = step * 64 + launch_id + 1`, where `step` is the epoch word the
  host increments after each step. A pair from an earlier step or launch therefore never looks current, so the
  scratch never needs to be cleared.
- A consumer loads the pairs it needs (batches of up to 12) and re-polls until all their tags are current, then uses
  the values. It waits only for its own inputs, not for the whole previous stage.
- Stores and loads use a device-scope cache policy, so producer and consumer meet in L2 rather than in a CU's
  private cache.

`scratch_layout` (in `layout.py`) gives every (layer, stage) its own region: e.g. `qp` (QSA projections), `ilog`
(index logits), `part` (attention partials), `mixx` (mixer outputs), `logit` (router), `mid` (expert activations),
`dpart` (down partials). Regions are only address bookkeeping. The traffic is what is written and read, and the
hand-off data mostly stays in L2 / MALL.

## 4. Guarded waits

A wait that never completes, because of a bug or a corrupted input, would hang the GPU. Every wait goes through
`common/guard.py`:

- A **deadline** of half the server's `--watchdog-timeout` (at most 60 s), checked on the 100 MHz `s_memrealtime`
  clock every 64 polls. The check uses only scalar registers, so it costs no vector registers at the stage's peak.
  The overhead is 0.1-0.5% of a step (sgl-project/sgl-amd-mono#19).
- On a timeout the wait sets a sticky error word and stops the launch with a store to address 0x0. A launch never
  runs on with values that did not arrive (their addresses would be garbage). The GPU fault stops the process, and
  no token of the failed step reaches a client.
- `SGLANG_AMD_MONO_FAULT_INJECT` stalls a chosen wait for the e2e test of this path.

## 5. Weights

`weights.py` reads SGLang's weights **in place** wherever the layout fits the kernel:

- MXFP4 experts: SGLang's FusedMoE `w13` / `w2` after AITER's weight processing, with the shared expert fused at
  index 512.
- W8A8 FP8 projections: Quark FP8 weights shuffled by AITER (`shuffle_weight(w, (16, 16))`) with f32 per-channel
  scales.

Copied (and logged at startup): the bf16 tensors the kernels need in their own MFMA tile order (HC mixers, router,
PLE projections, MTP input fusion), packed at build from the checkpoint files, and the GDN `in_proj_ba`, padded and
shuffled. The copies cost 2.45 GiB (target) + 0.54 GiB (draft) at k=3, counted before SGLang sizes the KV cache.

At startup a few experts and projections are compared byte for byte against the checkpoint. If an AITER layout change
breaks the assumptions, the server refuses to start instead of producing wrong tokens.

## 6. The MTP draft

One launch per draft step, same grid and mechanisms:

1. Input fusion: `fc_embedding(norm(embed(token)))` broadcast to the 4 branches plus `fc_hidden(norm(target residual
   streams))` per branch.
2. The draft's QSA layer and its MoE (same stages as above).
3. The draft's HC mixer; SGLang's logits processor samples the next draft token.

Extend runs R x L rows, each decode step R rows. SGLang's monokernel runtime routes only `DECODE` and
`TARGET_VERIFY` keys, so `draft.py` installs a small switch in front of the draft model's forward. It decides by
forward mode, batch size and row count only, so a captured CUDA graph replays the decision made at capture.

## 7. Host side and SGLang integration

- **Spec** (`spec.py`, `draft.py`): declares the model, the supported steps, device limits and `NUMERICS` bounds
  (`MonoSpec`); checks the config and refuses unsupported setups at startup with the reason.
- **Runner** (`gfx950/runner.py`): builds per-step pointer tables (one row per layer: weights, pools, cache rows; one
  global row: positions, KV locations, `req_to_token`, RoPE cache, mamba slot map, ...), cached by the addresses
  they hold so graph replays reuse them. It issues the 13 launches with the residual stream ping-ponging between two
  buffers and advances the epoch.
- **CUDA graphs**: every captured batch size 1-8 runs the launches. FlyDSL compiles each build at capture and keeps it in its disk cache, so a restart with the same pool layout captures in seconds.
- **No SGLang changes**: the plugin runs on unmodified SGLang main with `--monokernel-backend amd`
  (`--enable-amd-monokernel` after sgl-project/sglang#45389).

## 8. Correctness checks

- **Kernel test** (`test/kernels/test_qwen38_flash_next_launches.py`): 1-8 requests at k=3 and k=1 plus a padded
  batch, eager and replayed from CUDA graphs out of order. Every replay is bitwise equal to the first eager run (all
  outputs and pool tensors), and every step writes only its own rows.
- **Check mode** (`SGLANG_AMD_MONO_CHECK=1`): runs SGLang's forward ahead of every routed eager step on the same
  inputs and compares every output and pool region against the `NUMERICS` bounds (cosine and L2-relative per region;
  RoPE positions and n-gram context exact).
- **E2E** (`test/e2e/test_qwen38_flash_next_server.py`): routing of every graph, KL of decode against prefill, GSM8K
  across the 8-request switch point, check mode at k=3 and k=1, and the guarded-wait fault path.

## 9. Performance profile

8k/1k on MI355X, TP 1 (from the PR's report runs):

| k | concurrency | output tok/s, monokernel on | off | speed-up |
|---|---|---|---|---|
| 3 | 1 | 524.8 | 215.7 | 2.43x |
| 3 | 2 | 783.6 | 391.4 | 2.00x |
| 3 | 4 | 1075.0 | 658.3 | 1.63x |
| 3 | 8 | 1204.0 | 1057.3 | 1.14x |
| 1 | 8 | 1127.3 | 745.6 | 1.51x |

Where the time goes inside a launch of 3 GDN + 1 QSA layers (k=3; per-stage stamps of a `timeline=True` build):

| | R=1 (4 rows) | R=4 (16 rows) | R=8 (32 rows) |
|---|---|---|---|
| launch time | 364 us | 747 us | 1411 us |
| attention / GDN / mixers | 61% | 52% | 47% |
| router | 4% | 3% | 2% |
| MoE gate / up | 19% | 27% | 30% |
| MoE down (GEMM + partials) | 7% | 11% | 14% |
| MoE down combine | 6% | 5% | 5% |

As the batch grows, the union of routed experts covers more of the 512 experts (about half at 32 rows), so the MoE
weight stream grows and the gain over SGLang's kernels shrinks. This is why the switch point is 8 requests.

## 10. Limitations

- Model: Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8
- conc <= 8
- topk = 1
- TP = 1