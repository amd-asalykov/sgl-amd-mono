"""Qwen3.8-Flash-Next with --monokernel-backend amd on one MI355X GPU, NEXTN with
the MTP draft at k=3 and k=1 (RFC #43949), across the switch points: verify steps
of 1 to 8 requests and the draft's extend and decode steps run the launches, wider
ones the normal path, and requests join and leave a running batch that steps past
8 requests and back (a short GSM8K run, for gross failures); the tokens of covered
steps scored again by the normal prefill path (KL consistency); every routed step
against SGLang's own forward under SGLANG_AMD_MONO_CHECK, each output and pool
region within the spec's bounds; and a stalled wait that stops the server before
any token of the failed step is sent."""

import os
import re
import tempfile
import time
import unittest
from types import SimpleNamespace

import requests
from qwen38_flash_next_server import ENV, MODEL, args
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_amd_mono import BACKEND

LAUNCH_TIMEOUT = 3600
# The normal path scores about 0.97 on GSM8K (thinking on, low reasoning effort);
# the bound is for gross failures
GSM8K_BOUND = 0.90
# Decode against prefill, one batch of 1, 3, 7 and 8 prompts at a time: the normal path
# itself averages 0.02 to 0.09 a batch on these prompts (k=3 and k=1), the launches 0.015
# to 0.087
KL_BOUND = 0.12


def _launch(k, extra_env=None, extra_args=(), log=None):
    env = {**os.environ, **ENV, **(extra_env or {})}
    return popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=args(k) + ["--monokernel-backend", BACKEND, *extra_args],
        env=env,
        return_stdout_stderr=log,
    )


def _read(log) -> str:
    out, err = log
    out.seek(0)
    err.seek(0)
    return err.read() + out.read()


class _Serving:
    K = 3

    @classmethod
    def setUpClass(cls):
        cls.log = (
            tempfile.NamedTemporaryFile("w+", suffix=".out"),
            tempfile.NamedTemporaryFile("w+", suffix=".err"),
        )
        cls.process = _launch(
            cls.K, extra_args=["--watchdog-timeout", "600"], log=cls.log
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)
        for f in cls.log:
            f.close()

    def test_every_graph_up_to_8_requests_runs_the_launches(self):
        log = _read(self.log)
        m = re.search(
            r"runs target_verify steps of batch size \[([\d, ]+)\] x (\d+) tokens", log
        )
        self.assertIsNotNone(m, log[-3000:])
        # widths count rows: at k=1 the runtime routes up to 16 requests, and the
        # forward runs SGLang's for the 9 to 16 the launches do not take
        self.assertLessEqual(set(range(1, 9)), {int(b) for b in m[1].split(",")})
        self.assertEqual(int(m[2]), self.K + 1)
        self.assertIn(f"the launches run 1..8 requests x {self.K + 1} tokens", log)
        draft = set(re.findall(r"MTP draft runs \('(\w+)', (\d+), (\d+)\)", log))
        self.assertEqual(
            {int(b) for mode, b, _ in draft if mode == "DRAFT_EXTEND_V2"},
            set(range(1, 9)),
        )
        if self.K > 1:  # the draft's chain decode steps
            self.assertEqual(
                {int(b) for mode, b, _ in draft if mode == "DECODE"}, set(range(1, 9))
            )

    def test_covered_steps_match_prefill(self):
        # one graph per batch: 1, 3, 7 and 8 requests (pad rows at 3 and 7 in the
        # graphs of 4 and 8); with the cache flushed, no batch reads another's pages
        for rows in (1, 3, 7, 8):
            requests.post(DEFAULT_URL_FOR_TEST + "/flush_cache").raise_for_status()
            assert_logprobs_match(
                DEFAULT_URL_FOR_TEST,
                {MODEL: {"kl_div": KL_BOUND}},
                MODEL,
                max_samples=rows,
                max_new_tokens=512,
            )


class TestMonokernelK3(_Serving, CustomTestCase):
    K = 3

    def test_gsm8k_past_the_largest_width(self):
        # 12 requests at a time: the verify batch steps past 8 requests as it fills,
        # and back as it drains
        metrics = run_sgl_eval(
            SimpleNamespace(
                base_url=DEFAULT_URL_FOR_TEST,
                model=MODEL,
                eval_name="gsm8k",
                num_examples=200,
                num_threads=12,
                max_tokens=16384,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                seed=42,
                sgl_eval_thinking=True,
                chat_template_kwargs={"reasoning_effort": "low"},
            )
        )
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], GSM8K_BOUND)


class TestMonokernelK1(_Serving, CustomTestCase):
    K = 1


class TestMonokernelCheck(CustomTestCase):
    def _check(self, k):
        # SGLANG_AMD_MONO_CHECK (eager) runs SGLang's forward ahead of each routed
        # step: 1, 3 and 8 requests (12 runs the normal path), and a long prompt
        # whose index keys compress across many groups
        para = "The river valley archive lists the flood height of every year. "
        log = (
            tempfile.NamedTemporaryFile("w+", suffix=".out"),
            tempfile.NamedTemporaryFile("w+", suffix=".err"),
        )
        process = _launch(
            k,
            extra_env={"SGLANG_AMD_MONO_CHECK": "1"},
            extra_args=["--disable-cuda-graph", "--watchdog-timeout", "600"],
            log=log,
        )
        try:
            for n, repeats in ((1, 2), (3, 2), (8, 2), (12, 2), (1, 120)):
                r = requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "text": [f"Record {i}. " + para * repeats for i in range(n)],
                        "sampling_params": {"max_new_tokens": 32, "temperature": 0},
                    },
                    timeout=1800,
                )
                r.raise_for_status()
        finally:
            terminate_and_kill_process_tree(process)
        text = _read(log)
        for f in log:
            f.close()
        checks = re.findall(
            r"monokernel check (verify|draft \w+) (\d+) x (\d+) \(argmax agrees on ([\d.]+)\): (.*)",
            text,
        )
        values = {}
        for *_, fields in checks:
            for name, v in re.findall(r"(\S+)=(\S+)", fields):
                values.setdefault(name, []).append(float(v))
        for name, vs in sorted(values.items()):
            print(
                f"k={k} {name}: min {min(vs):.6g} max {max(vs):.6g} ({len(vs)} checks)"
            )
        seen = {(kind, int(b)) for kind, b, _, _, _ in checks}
        want = {
            (kind, b) for kind in ("verify", "draft DRAFT_EXTEND_V2") for b in (1, 3, 8)
        }
        if k > 1:
            want |= {("draft DECODE", b) for b in (1, 3, 8)}
        self.assertTrue(want <= seen, seen)
        self.assertFalse(any(kind == "verify" and int(b) > 8 for kind, b, *_ in checks))
        self.assertGreater(
            sum(values.get("verify.ckv.rows", [0])), 0
        )  # keys compressed
        violations = re.findall(r"monokernel check violation: .*", text)
        self.assertEqual(violations, [], "\n".join(violations[:20]))

    def test_k3_steps_within_their_numerics(self):
        self._check(3)

    def test_k1_steps_within_their_numerics(self):
        self._check(1)


class TestMonokernelFault(CustomTestCase):
    def test_a_stalled_wait_stops_the_server_without_tokens(self):
        # past the graph capture and warmup (1000 steps of a runner: the draft's within
        # the request's 2 x 1024 tokens), the first wait of the launch stalls; at the
        # deadline (half the watchdog) it stops the launch, and SGLang the server
        log = (
            tempfile.NamedTemporaryFile("w+", suffix=".out"),
            tempfile.NamedTemporaryFile("w+", suffix=".err"),
        )
        process = _launch(
            3,
            # the GPU fault's core dump (all of the GPU's memory) would keep the
            # server alive for minutes
            extra_env={
                "SGLANG_AMD_MONO_FAULT_INJECT": "0:1:1000",
                "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
            },
            extra_args=["--watchdog-timeout", "20"],
            log=log,
        )
        try:
            texts = []
            try:
                r = requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "text": ["Write a long essay about lighthouses."] * 2,
                        "sampling_params": {
                            "max_new_tokens": 1024,
                            "temperature": 0,
                            "ignore_eos": True,
                        },
                    },
                    timeout=300,
                )
                if r.status_code == 200:
                    texts.extend(item.get("text") for item in r.json())
            except requests.RequestException:
                pass
            self.assertFalse(any(texts), "a failed step's tokens reached the client")
            deadline = time.time() + 300
            while process.poll() is None and time.time() < deadline:
                time.sleep(2)
            self.assertIsNotNone(process.poll(), "the server kept running")
            text = _read(log)
            # a launch never runs on after a failed wait: it stops with a store to
            # the null page, so the only faults are at address 0x0
            faults = re.findall(r"faulting addr: (0x[0-9a-fA-F]+)", text)
            self.assertTrue(faults, text[-2000:])
            self.assertEqual(set(faults) - {"0x0"}, set(), faults)
        finally:
            terminate_and_kill_process_tree(process)
            for f in log:
                f.close()


if __name__ == "__main__":
    unittest.main()
