"""Tests for device detection and the LS_DEVICE override."""

import os
from unittest import mock

from django.test import TestCase

from latent_search.server.indexing.services import device as device_mod


class DeviceOverrideTest(TestCase):
    """Verify LS_DEVICE env var overrides auto-detection."""

    def setUp(self) -> None:
        # Reset the cached device so each test re-evaluates.
        device_mod._DEVICE = None

    def tearDown(self) -> None:
        device_mod._DEVICE = None

    def test_override_cpu_forces_cpu_even_with_cuda(self) -> None:
        """LS_DEVICE=cpu wins even if torch.cuda.is_available() is True."""
        with mock.patch.object(device_mod.torch, "cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            with mock.patch.dict(os.environ, {"LS_DEVICE": "cpu"}):
                self.assertEqual(device_mod.get_device(), "cpu")

    def test_override_cuda_forces_cuda_without_checking_availability(self) -> None:
        """LS_DEVICE=cuda is honoured without calling is_available()."""
        with mock.patch.object(device_mod.torch, "cuda") as mock_cuda:
            mock_cuda.is_available.return_value = False
            with mock.patch.dict(os.environ, {"LS_DEVICE": "cuda"}):
                self.assertEqual(device_mod.get_device(), "cuda")
            # is_available should not have been consulted
            mock_cuda.is_available.assert_not_called()

    def test_no_override_falls_back_to_autodetect(self) -> None:
        with mock.patch.object(device_mod.torch, "cuda") as mock_cuda:
            mock_cuda.is_available.return_value = False
            # Ensure LS_DEVICE is unset
            env = {k: v for k, v in os.environ.items() if k != "LS_DEVICE"}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(device_mod.get_device(), "cpu")

    def test_invalid_override_falls_back_to_autodetect(self) -> None:
        with mock.patch.object(device_mod.torch, "cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            with mock.patch.dict(os.environ, {"LS_DEVICE": "tpu"}):
                self.assertEqual(device_mod.get_device(), "cuda")
