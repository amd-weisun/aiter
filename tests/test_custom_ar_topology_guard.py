# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the custom allreduce cross-socket PCIe topology guard."""

import importlib.util
import os
import pathlib
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

_REPO = pathlib.Path(__file__).resolve().parents[1]
_GUARD = _REPO / "aiter" / "dist" / "device_communicators" / "_topology_guard.py"


def _load_guard():
    # Load the policy module without importing AITER's GPU operators.
    spec = importlib.util.spec_from_file_location("_test_topology_guard", _GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 8 GPUs, 4 per socket, as on a 2-socket host.
_BDFS = [f"0000:{b}:00.0" for b in ("05", "08", "75", "78", "85", "88", "f5", "f8")]
_TWO_SOCKETS = [i // 4 for i in range(8)]


class CrossSocketPcieReasonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = _load_guard()

    def reason(self, bdfs, numas, xgmi):
        calls = []

        def query(b):
            calls.append(list(b))
            return xgmi

        return self.guard.cross_socket_pcie_reason(bdfs, numas, query), calls

    def test_xgmi_two_sockets_keeps_custom_ar(self):
        # MI300X / MI355X: two sockets, but every pair is 1-hop xGMI.
        reason, calls = self.reason(_BDFS, _TWO_SOCKETS, True)
        self.assertIsNone(reason)
        self.assertEqual(calls, [_BDFS])

    def test_pcie_two_sockets_disables(self):
        reason, _ = self.reason(_BDFS, _TWO_SOCKETS, False)
        self.assertIsNotNone(reason)
        self.assertIn("NUMA nodes [0, 1]", reason)

    def test_pcie_four_gpus_across_sockets_disables(self):
        bdfs = [_BDFS[i] for i in (0, 1, 4, 5)]
        reason, _ = self.reason(bdfs, [0, 0, 1, 1], False)
        self.assertIsNotNone(reason)

    def test_single_socket_keeps_and_skips_link_query(self):
        reason, calls = self.reason(_BDFS[:4], [0, 0, 0, 0], False)
        self.assertIsNone(reason)
        self.assertEqual(calls, [])

    def test_link_query_unknown_keeps(self):
        reason, _ = self.reason(_BDFS, _TWO_SOCKETS, None)
        self.assertIsNone(reason)

    def test_unknown_numa_keeps(self):
        reason, calls = self.reason(_BDFS, [0] * 7 + [None], False)
        self.assertIsNone(reason)
        self.assertEqual(calls, [])

    def test_unknown_bdf_keeps(self):
        reason, calls = self.reason([None] + _BDFS[1:], _TWO_SOCKETS, False)
        self.assertIsNone(reason)
        self.assertEqual(calls, [])

    def test_empty_group_keeps(self):
        reason, _ = self.reason([], [], False)
        self.assertIsNone(reason)


class HelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = _load_guard()

    def test_device_bdf_formats_pci_address(self):
        props = SimpleNamespace(pci_domain_id=0, pci_bus_id=0xF5, pci_device_id=0)
        self.assertEqual(self.guard.device_bdf(props), "0000:f5:00.0")

    def test_device_bdf_missing_fields(self):
        self.assertIsNone(self.guard.device_bdf(SimpleNamespace()))

    def test_read_numa_node(self):
        with tempfile.TemporaryDirectory() as root:
            for bdf, value in (
                ("0000:05:00.0", "1\n"),
                ("0000:08:00.0", "-1\n"),
                ("0000:75:00.0", "x"),
            ):
                os.makedirs(os.path.join(root, bdf))
                with open(os.path.join(root, bdf, "numa_node"), "w") as f:
                    f.write(value)
            self.assertEqual(self.guard.read_numa_node("0000:05:00.0", root), 1)
            self.assertIsNone(self.guard.read_numa_node("0000:08:00.0", root))  # -1
            self.assertIsNone(
                self.guard.read_numa_node("0000:75:00.0", root)
            )  # garbage
            self.assertIsNone(
                self.guard.read_numa_node("0000:aa:00.0", root)
            )  # missing
            self.assertIsNone(self.guard.read_numa_node(None, root))

    def _fake_amdsmi(self, link_type_fn, init_raises=False):
        m = types.ModuleType("amdsmi")
        m.amdsmi_init = mock.Mock(
            side_effect=RuntimeError("no amdsmi") if init_raises else None
        )
        m.amdsmi_shut_down = mock.Mock()
        m.amdsmi_get_processor_handle_from_bdf = lambda bdf: bdf
        m.amdsmi_topo_get_link_type = link_type_fn
        return m

    def test_amdsmi_all_xgmi(self):
        m = self._fake_amdsmi(lambda a, b: {"type": 2, "hops": 1})
        with mock.patch.dict(sys.modules, {"amdsmi": m}):
            self.assertTrue(self.guard.amdsmi_all_xgmi_one_hop(_BDFS))
        m.amdsmi_shut_down.assert_called_once()

    def test_amdsmi_one_pcie_pair(self):
        def link(a, b):
            return {"type": 1 if (a, b) == (_BDFS[0], _BDFS[7]) else 2, "hops": 1}

        m = self._fake_amdsmi(link)
        with mock.patch.dict(sys.modules, {"amdsmi": m}):
            self.assertFalse(self.guard.amdsmi_all_xgmi_one_hop(_BDFS))
        m.amdsmi_shut_down.assert_called_once()

    def test_amdsmi_enum_link_type(self):
        enum_xgmi = SimpleNamespace(value=2)
        m = self._fake_amdsmi(lambda a, b: {"type": enum_xgmi, "hops": 1})
        with mock.patch.dict(sys.modules, {"amdsmi": m}):
            self.assertTrue(self.guard.amdsmi_all_xgmi_one_hop(_BDFS[:2]))

    def test_amdsmi_failures_are_unknown(self):
        def boom(a, b):
            raise RuntimeError("query failed")

        with mock.patch.dict(sys.modules, {"amdsmi": self._fake_amdsmi(boom)}):
            self.assertIsNone(self.guard.amdsmi_all_xgmi_one_hop(_BDFS))
        with mock.patch.dict(
            sys.modules, {"amdsmi": self._fake_amdsmi(boom, init_raises=True)}
        ):
            self.assertIsNone(self.guard.amdsmi_all_xgmi_one_hop(_BDFS))
        with mock.patch.dict(sys.modules, {"amdsmi": None}):  # import fails
            self.assertIsNone(self.guard.amdsmi_all_xgmi_one_hop(_BDFS))


if __name__ == "__main__":
    unittest.main()
