"""The Qwen3.8-Flash-Next gfx950 launches on one GPU (RFC #43949), on a 12-layer
prefix of the checkpoint (every launch kind runs): the target verify step (13
launches) of 1..8 requests at 2 and 4 rows each, a batch padded as SGLang pads a
graph's, and the MTP draft's extend and decode steps, eagerly and replayed from CUDA
graphs captured at several widths out of order. A second eager run and every replay
equal the first eager run bitwise (every output and pool tensor), and every step
writes only its own rows (``qwen38_flash_next/check_launches.py``).

The checkpoint is QWEN38_FLASH_NEXT_MODEL_PATH, a directory or a Hugging Face id
(default amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8; only the shards the
fixtures read are downloaded); the first run compiles about 100 kernels (16 at a
time)."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch

CHECK = Path(__file__).with_name("qwen38_flash_next") / "check_launches.py"


def _gfx950() -> bool:
    return bool(
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    )


@unittest.skipUnless(_gfx950(), "needs a gfx950 GPU")
class TestQwen38FlashNextLaunches(unittest.TestCase):
    def _check(self, *args: str) -> None:
        result = subprocess.run([sys.executable, str(CHECK), *args], timeout=7200)
        self.assertEqual(result.returncode, 0, " ".join(args))

    def test_target_verify_launches(self):
        self._check("--what", "target", "--R", "1-8", "--L", "2,4", "--graph")

    def test_mtp_draft_launches(self):
        self._check("--what", "draft", "--R", "1-8", "--L", "2,4", "--graph")


if __name__ == "__main__":
    unittest.main()
