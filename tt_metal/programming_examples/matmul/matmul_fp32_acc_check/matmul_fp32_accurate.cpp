// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Every variant of the FP32-accurate matmul, run at once, reported by the device.
//
// USEFUL_PER_SOP 1 through 8 and the LLK matmul each get a core, and an aggregator core collects
// their output tiles and prints the table. The host only launches: the operands, the reference
// and the error bound are constant expressions the kernels evaluate themselves, so nothing about
// the problem crosses the PCIe link in either direction.
//
// Running them together is what makes the table meaningful. Nine separate launches would leave
// the reader to take on trust that the conditions matched, and could not compare two variants
// against each other at all - which is how the aggregator establishes that S=8 and the LLK
// matmul are the same computation, differing only in the order the partial sums reach Dst.
//
// Needs TT_METAL_DPRINT_CORES=all, and silicon: ttsim has no device print buffer.

#include <tt-metalium/distributed.hpp>

#include "runner.hpp"

int main() {
    using namespace mm_host;

    constexpr int device_id = 0;
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

    print_terms(problem::Layout::Packed);
    run_all_variants(
        problem::kM, problem::kN, problem::k_dim(problem::Layout::Packed), mesh_device, problem::Layout::Packed);

    const bool closed = mesh_device->close();
    fmt::print("check=device_closed expected=1 actual={} result={}\n", closed, okng(closed));
    return closed ? 0 : 1;
}
