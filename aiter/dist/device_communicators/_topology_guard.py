# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Topology guard for custom allreduce on PCIe-only, multi-socket hosts.

Custom allreduce runs full-mesh peer-to-peer kernels. On GPUs connected only by
PCIe whose TP group spans more than one CPU socket (NUMA node), the 2-stage
kernel has frozen and rebooted the whole host (2-socket EPYC + 8x MI350P,
world size 8). This module decides, from data the caller gathers from every
rank, whether custom allreduce must be disabled for such a group.

Policy: disable only when the group is *known* to span NUMA nodes AND amdsmi
*positively* reports a GPU pair that is not 1-hop xGMI. xGMI-connected groups
(MI300X / MI355X, including 2-socket nodes), single-socket groups and every
"unknown" case keep the existing behavior, so a detection failure can never
regress a working system.

Kept free of torch / amdsmi imports at module level so the policy can be unit
tested on a CPU-only machine.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence

ALLOW_CROSS_SOCKET_PCIE_ENV = "AITER_CUSTOM_AR_ALLOW_CROSS_SOCKET_PCIE"
AMDSMI_LINK_TYPE_XGMI = 2

_SYSFS_PCI_DEVICES = "/sys/bus/pci/devices"


def device_bdf(props) -> str | None:
    """PCI address ``dddd:bb:dd.0`` from torch device properties, or None."""
    try:
        return (
            f"{int(props.pci_domain_id):04x}:{int(props.pci_bus_id):02x}:"
            f"{int(props.pci_device_id):02x}.0"
        )
    except (AttributeError, TypeError, ValueError):
        return None


def read_numa_node(bdf: str | None, sysfs_root: str = _SYSFS_PCI_DEVICES) -> int | None:
    """NUMA node of a PCI device from sysfs, or None if unknown (-1 or unreadable)."""
    if bdf is None:
        return None
    try:
        with open(f"{sysfs_root}/{bdf}/numa_node") as f:
            node = int(f.read().strip())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


def amdsmi_all_xgmi_one_hop(bdfs: Sequence[str]) -> bool | None:
    """True if every GPU pair is 1-hop xGMI, False if any pair is not, None if unknown."""
    try:
        import amdsmi
    except Exception:  # noqa: BLE001 - amdsmi is optional
        return None
    try:
        amdsmi.amdsmi_init()
    except Exception:  # noqa: BLE001
        return None
    try:
        handles = [amdsmi.amdsmi_get_processor_handle_from_bdf(b) for b in bdfs]
        for i in range(len(handles)):
            for j in range(i + 1, len(handles)):
                link = amdsmi.amdsmi_topo_get_link_type(handles[i], handles[j])
                link_type = int(getattr(link["type"], "value", link["type"]))
                if link_type != AMDSMI_LINK_TYPE_XGMI or int(link["hops"]) != 1:
                    return False
        return True
    except Exception:  # noqa: BLE001
        return None
    finally:
        with contextlib.suppress(Exception):
            amdsmi.amdsmi_shut_down()


def cross_socket_pcie_reason(
    bdfs: Sequence[str | None],
    numa_nodes: Sequence[int | None],
    all_xgmi_one_hop: Callable[[Sequence[str]], bool | None] = amdsmi_all_xgmi_one_hop,
) -> str | None:
    """Why custom allreduce must be disabled for this group, or None to keep it.

    ``bdfs`` / ``numa_nodes`` hold one entry per rank. ``all_xgmi_one_hop`` is
    only called for groups known to span NUMA nodes.
    """
    if not bdfs or None in bdfs or None in numa_nodes:
        return None
    nodes = sorted(set(numa_nodes))
    if len(nodes) <= 1:
        return None
    if all_xgmi_one_hop(list(bdfs)) is not False:
        return None
    return (
        f"GPUs {list(bdfs)} are not xGMI fully connected (PCIe) and span "
        f"NUMA nodes {nodes}; cross-socket peer-to-peer allreduce can hang "
        f"the host"
    )
