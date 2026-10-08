from __future__ import annotations

import io
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from soar.engine.trainer import BaseTrainer
from soar.utils.device import announce_device, resolve_device


class DeviceSelectionTests(unittest.TestCase):
    @staticmethod
    def _announce(selection) -> str:
        stream = io.StringIO()
        with redirect_stdout(stream):
            announce_device(selection)
        return stream.getvalue()

    def test_cuda_device_is_announced(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "get_device_name", return_value="Test GPU"),
            patch.object(torch.version, "cuda", "12.4"),
        ):
            selection = resolve_device("cuda")
            output = self._announce(selection)

        self.assertEqual(selection.device.type, "cuda")
        self.assertFalse(selection.fell_back)
        self.assertIn("Using CUDA GPU: Test GPU", output)
        self.assertIn("CUDA runtime 12.4", output)

    def test_explicit_cpu_is_announced_without_fallback(self):
        with patch.object(torch.cuda, "is_available", return_value=True):
            selection = resolve_device("cpu")
            output = self._announce(selection)

        self.assertEqual(selection.device.type, "cpu")
        self.assertFalse(selection.fell_back)
        self.assertIn("Using CPU (explicitly requested)", output)
        self.assertNotIn("falling back", output)

    def test_unavailable_cuda_falls_back_with_warning(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            selection = resolve_device("cuda:0")
            output = self._announce(selection)

        self.assertEqual(selection.device.type, "cpu")
        self.assertTrue(selection.fell_back)
        self.assertIn("CUDA was requested but is unavailable", output)
        self.assertIn("falling back to CPU", output)

    def test_invalid_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported device"):
            resolve_device("gpu")

    def test_unavailable_cuda_index_is_rejected(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=1),
        ):
            with self.assertRaisesRegex(ValueError, "CUDA device index 2 is unavailable"):
                resolve_device("cuda:2")

    def test_trainer_announces_cuda_fallback_at_startup(self):
        dataset_config = SimpleNamespace(root_path=Path("."), nc=1, names={0: "class_0"})
        setup_methods = (
            "_setup_directories",
            "_setup_model",
            "_setup_optimizer",
            "_setup_scheduler",
            "_setup_loss",
            "_setup_data",
            "_setup_ema",
            "_setup_validator",
            "_setup_amp",
            "_load_checkpoint",
        )

        stream = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(
                patch("soar.engine.trainer.DatasetConfig.resolve", return_value=dataset_config)
            )
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            for method in setup_methods:
                stack.enter_context(patch.object(BaseTrainer, method, return_value=None))
            stack.enter_context(redirect_stdout(stream))
            trainer = BaseTrainer(model_cfg="soar", data_root="unused", device="cuda")

        self.assertEqual(trainer.device.type, "cpu")
        self.assertIn("falling back to CPU", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
