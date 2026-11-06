"""Focused remote-only tests for Step30 operational safety contracts."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from step30_remote_safety_v1 import (
    RemoteSafetyError,
    active_step30_processes,
    build_input_inventory,
    ensure_distinct_paths,
    ensure_remote_server,
    scheduler_cpu_capacity,
    validate_state_root,
)


class RemoteSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.original)

    def test_direct_remote_requires_explicit_opt_in(self) -> None:
        os.environ["STEP30_REMOTE_EXECUTION"] = "1"
        os.environ.pop("STEP30_ALLOW_DIRECT_REMOTE", None)
        with self.assertRaises(RemoteSafetyError):
            ensure_remote_server(target="dcu_cpu", allow_direct_remote=True)

    def test_direct_remote_uses_visible_cpu_affinity(self) -> None:
        os.environ["STEP30_REMOTE_EXECUTION"] = "1"
        os.environ["STEP30_ALLOW_DIRECT_REMOTE"] = "1"
        info = ensure_remote_server(target="dcu_cpu", allow_direct_remote=True)
        self.assertEqual(info["scheduler"], "direct_remote")
        self.assertGreater(scheduler_cpu_capacity(allow_direct_remote=True), 0)

    def test_existing_state_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "existing"
            root.mkdir()
            with self.assertRaises(RemoteSafetyError):
                validate_state_root(root, resume=False)

    def test_nested_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(RemoteSafetyError):
                ensure_distinct_paths({"state": root / "state", "artifact": root})

    def test_input_inventory_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.txt"
            source.write_text("step30\n", encoding="utf-8")
            first = build_input_inventory([("source", source, "file")])
            second = build_input_inventory([("source", source, "file")])
            self.assertEqual(first, second)

    def test_active_step30_process_scan_returns_list(self) -> None:
        self.assertIsInstance(active_step30_processes(), list)


if __name__ == "__main__":
    unittest.main()

