// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The 8 x 4 torus of the Galaxy as the Ethernet links see it (worklog 14 I-0) and the BFS trees of a 2 x 4 block
// (comm_model.py Mesh.bfs_tree: vertical first, then the shorter ring direction), shared by the dataflow tests.
#pragma once
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <vector>
#include <tt-metalium/core_coord.hpp>
#include <llrt/tt_cluster.hpp>
#include "impl/context/metal_context.hpp"

namespace qwen38_dataflow {

struct Torus {
    int row[32];
    int col[32];
    int at[8][4];
    std::map<std::pair<int, int>, std::vector<tt::tt_metal::CoreCoord>>
        link_cores;  // (chip, peer) -> logical eth cores of chip
    std::map<std::pair<int, tt::tt_metal::CoreCoord>, std::pair<int, tt::tt_metal::CoreCoord>>
        peer;  // (chip, core) -> (peer chip, peer core)
};

// the torus position of every chip from the links (README 12.5 / worklog 14 I-0): 4-cycles classify the edges --
// an edge on three 4-cycles runs along the 4-long ring, on two along the 8-long one
inline Torus map_torus() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    Torus t{};
    std::map<int, std::set<int>> adj;
    for (const auto& [chip, chans] : cluster.get_ethernet_connections()) {
        for (const auto& [chan, remote] : chans) {
            const auto& [rchip, rchan] = remote;
            if (rchip == chip) {
                continue;
            }
            const auto core = cluster.get_soc_desc(chip).get_eth_core_for_channel(chan, tt::CoordSystem::LOGICAL);
            const auto rcore = cluster.get_soc_desc(rchip).get_eth_core_for_channel(rchan, tt::CoordSystem::LOGICAL);
            adj[chip].insert(rchip);
            t.link_cores[{chip, rchip}].push_back(tt::tt_metal::CoreCoord(core.x, core.y));
            t.peer[{chip, tt::tt_metal::CoreCoord(core.x, core.y)}] = {
                rchip, tt::tt_metal::CoreCoord(rcore.x, rcore.y)};
        }
    }
    TT_FATAL(adj.size() == 32, "expected 32 chips, got {}", adj.size());
    std::map<std::pair<int, int>, int> cycles;
    std::set<std::set<int>> seen;
    for (auto& [a, na] : adj) {
        for (int b : na) {
            for (int c : adj[b]) {
                if (c == a) {
                    continue;
                }
                for (int d : adj[c]) {
                    if (d == a || d == b || !adj[d].count(a)) {
                        continue;
                    }
                    std::set<int> key{a, b, c, d};
                    if (!seen.insert(key).second) {
                        continue;
                    }
                    const int ring[4] = {a, b, c, d};
                    for (int i = 0; i < 4; ++i) {
                        const int u = ring[i], v = ring[(i + 1) % 4];
                        ++cycles[{std::min(u, v), std::max(u, v)}];
                    }
                }
            }
        }
    }
    std::map<int, std::set<int>> along4, along8;
    for (auto& [e, n] : cycles) {
        (n == 3 ? along4 : along8)[e.first].insert(e.second);
        (n == 3 ? along4 : along8)[e.second].insert(e.first);
    }
    std::vector<int> column{0};
    int prev = -1;
    while (column.size() < 8) {
        for (int v : along8[column.back()]) {
            if (v != prev && std::find(column.begin(), column.end(), v) == column.end()) {
                prev = column.back();
                column.push_back(v);
                break;
            }
        }
    }
    for (int i = 0; i < 8; ++i) {
        int row4[4] = {column[i], -1, -1, -1};
        if (i == 0) {
            row4[1] = *along4[column[0]].begin();
        } else {
            for (int v : along4[column[i]]) {
                if (adj[v].count(t.at[i - 1][1])) {
                    row4[1] = v;
                }
            }
        }
        for (int j = 2; j < 4; ++j) {
            for (int v : along4[row4[j - 1]]) {
                if (v != row4[j - 2]) {
                    row4[j] = v;
                }
            }
        }
        for (int j = 0; j < 4; ++j) {
            t.at[i][j] = row4[j];
            t.row[row4[j]] = i;
            t.col[row4[j]] = j;
        }
    }
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            TT_FATAL(
                adj[t.at[i][j]].count(t.at[i][(j + 1) % 4]) && adj[t.at[i][j]].count(t.at[(i + 1) % 8][j]),
                "not an 8 x 4 torus at ({}, {})",
                i,
                j);
        }
    }
    return t;
}

// the BFS tree of a 2 x 4 block rooted at local position r (row 0/1, column 0..3): vertical first, then the shorter
// ring direction (comm_model.py Mesh.bfs_tree); returns parent of every node (-1 for the root)
inline std::array<int, 8> bfs_parent(int root) {
    std::array<int, 8> parent{};
    const int rr = root / 4, rc = root % 4;
    for (int n = 0; n < 8; ++n) {
        if (n == root) {
            parent[n] = -1;
            continue;
        }
        const int nr = n / 4, nc = n % 4;
        int dc = nc - rc;
        if (dc > 2) {
            dc -= 4;
        }
        if (dc < -2) {
            dc += 4;
        }
        std::vector<int> path;
        int cr = rr, cc = rc;
        if (nr != rr) {
            cr = nr;
            path.push_back(cr * 4 + cc);
        }
        while (dc != 0) {
            const int s = dc > 0 ? 1 : -1;
            cc = (cc + s + 4) % 4;
            dc -= s;
            path.push_back(cr * 4 + cc);
        }
        parent[n] = path.size() >= 2 ? path[path.size() - 2] : root;
    }
    return parent;
}

}  // namespace qwen38_dataflow
