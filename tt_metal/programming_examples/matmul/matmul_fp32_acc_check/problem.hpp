// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The problem this example poses, as constant expressions shared by the host and the kernels.
//
// Everything here - the operands, their tile layout, the reference result and the error bound -
// is a function of compile-time constants, so none of it has to be computed at runtime or moved
// across the PCIe link. The reader materialises the operands straight into L1, and the writer
// checks the output in place.
//
// Note the BF16 rounding: the device only ever sees BF16 datums, so the reference is built from
// those and not from the pre-rounding floats. Comparing against the floats would fold input
// quantization into what is meant to be a measurement of accumulator behaviour.

#pragma once

#include <cstdint>

namespace mm_fp32_acc_check {

constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kTileWidth = 32;
constexpr uint32_t kFaceDim = 16;

// Output is a single 32x32 tile; only K varies between the layouts under test.
constexpr uint32_t kM = kTileHeight;
constexpr uint32_t kN = kTileWidth;

// Each MVMUL reduces 16 K-elements as two independent 8-lane SOP groups.
constexpr uint32_t kSopGroupLanes = 8;
constexpr uint32_t kNumValues = 32;

// The three K-layouts. Only the placement of the same 32 values differs.
enum class Layout : uint32_t {
    Packed,   // K=32, all eight values of a group share it
    Spread,   // K=256, one value per group
    Uniform,  // K=32, group full but all its values share one exponent
};

constexpr uint32_t k_dim(Layout layout) {
    return (layout == Layout::Spread) ? (kNumValues * kSopGroupLanes) : kNumValues;
}

// ---------------------------------------------------------------------------
// Minimal constexpr float helpers.
//
// <cmath> is not usable in a kernel, and these are all the math the problem needs. Exponents are
// small and exact here, so a plain loop is both correct and constant-foldable.
// ---------------------------------------------------------------------------

constexpr float const_ldexp(float x, int e) {
    for (; e > 0; --e) {
        x *= 2.0f;
    }
    for (; e < 0; ++e) {
        x *= 0.5f;
    }
    return x;
}

constexpr float const_fabs(float x) { return x < 0.0f ? -x : x; }
constexpr double const_fabs(double x) { return x < 0.0 ? -x : x; }

// Round-to-nearest-even from FP32 to BF16, returned as the raw 16-bit datum.
constexpr uint16_t to_bf16_bits(float x) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, x);
    const uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7FFFu + lsb) >> 16);
}

constexpr float from_bf16_bits(uint16_t bits) {
    return __builtin_bit_cast(float, static_cast<uint32_t>(bits) << 16);
}

// The value a BF16 round-trip leaves behind, which is what the device actually multiplies.
constexpr float bf16(float x) { return from_bf16_bits(to_bf16_bits(x)); }

// ---------------------------------------------------------------------------
// Operands
//
// A[m][k] = (1 + m/M) * terms[k],  B[k][n] = 1 + n/N
// so exact C[m][n] = (1 + m/M) * (1 + n/N) * sum_k terms[k].
//
// Both operands vary along every axis, so a transposed or face-swapped operand cannot go
// unnoticed, and every output element is a distinct value - which is what makes the comparison
// meaningful across the whole tile rather than at element 0 only.
// ---------------------------------------------------------------------------

// The 32 values under test: exponents spanning ~21 bits, in eight steps of 3 binades.
constexpr float value_at(uint32_t j) {
    return const_ldexp(1.0f + 0.125f * static_cast<float>(j % 8), -3 * static_cast<int>(j % 8));
}

// terms[k] for a layout: Packed takes the values as they are, Spread puts one per SOP group with
// zeros between, and Uniform gives every value of a group the same exponent.
constexpr float term_at(Layout layout, uint32_t k) {
    if (layout == Layout::Spread) {
        return (k % kSopGroupLanes == 0) ? value_at(k / kSopGroupLanes) : 0.0f;
    }
    if (layout == Layout::Uniform) {
        return const_ldexp(1.0f, -3 * static_cast<int>(k / kSopGroupLanes));
    }
    return value_at(k);
}

constexpr float a_at(Layout layout, uint32_t m, uint32_t k) {
    const float row_scale = 1.0f + static_cast<float>(m) / static_cast<float>(kM);
    return bf16(row_scale * term_at(layout, k));
}

constexpr float b_at(uint32_t n) { return bf16(1.0f + static_cast<float>(n) / static_cast<float>(kN)); }

// ---------------------------------------------------------------------------
// Tile layout
//
// TILED_NFACES: a 32x32 tile is four 16x16 faces in order TL, TR, BL, BR, each stored row-major.
// A matrix wider or taller than one tile is a row-major grid of such tiles.
// ---------------------------------------------------------------------------

constexpr uint32_t tiled_index(uint32_t row, uint32_t col, uint32_t cols) {
    const uint32_t tile_r = row / kTileHeight;
    const uint32_t tile_c = col / kTileWidth;
    const uint32_t tiles_per_row = cols / kTileWidth;
    const uint32_t tile_base = (tile_r * tiles_per_row + tile_c) * (kTileHeight * kTileWidth);

    const uint32_t r = row % kTileHeight;
    const uint32_t c = col % kTileWidth;
    const uint32_t face = (r / kFaceDim) * 2 + (c / kFaceDim);
    return tile_base + face * (kFaceDim * kFaceDim) + (r % kFaceDim) * kFaceDim + (c % kFaceDim);
}

// ---------------------------------------------------------------------------
// Reference and error bound
// ---------------------------------------------------------------------------

// Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., section 3.1: a length-n inner
// product accumulated with unit roundoff u satisfies
//   |computed - exact| <= gamma_n * sum_k |a_k * b_k|,  gamma_n = n*u / (1 - n*u).
// With an FP32 accumulator u = 2^-24. This is the bound a correct FP32-accumulating dot product
// must respect; it replaces the hand-picked "effective bits" thresholds this example used before,
// which had no derivation behind them.
constexpr double gamma_n(uint32_t n) {
    constexpr double u = 1.0 / 16777216.0;  // 2^-24
    const double nu = static_cast<double>(n) * u;
    return nu / (1.0 - nu);
}

// B[k][n] does not depend on k, so both sums below factor into a row sum times b_at(n). Keeping
// them in that form matters for the kernels: computed per element the K loop runs kM*kN times,
// and under ttsim, which simulates instruction by instruction, that is the difference between
// seconds and tens of minutes.
constexpr double row_sum(Layout layout, uint32_t m) {
    double acc = 0.0;
    for (uint32_t k = 0; k < k_dim(layout); ++k) {
        acc += static_cast<double>(a_at(layout, m, k));
    }
    return acc;
}

constexpr double row_abs_sum(Layout layout, uint32_t m) {
    double acc = 0.0;
    for (uint32_t k = 0; k < k_dim(layout); ++k) {
        acc += const_fabs(static_cast<double>(a_at(layout, m, k)));
    }
    return acc;
}

// Exact C[m][n], from the BF16 datums the device receives.
constexpr double expected_at(Layout layout, uint32_t m, uint32_t n) {
    return row_sum(layout, m) * static_cast<double>(b_at(n));
}

// sum_k |a_k * b_k| for the same element, which is what the bound scales.
constexpr double abs_sum_at(Layout layout, uint32_t m, uint32_t n) {
    return row_abs_sum(layout, m) * const_fabs(static_cast<double>(b_at(n)));
}

constexpr double bound_at(Layout layout, uint32_t m, uint32_t n) {
    return gamma_n(k_dim(layout)) * abs_sum_at(layout, m, n);
}

}  // namespace mm_fp32_acc_check
