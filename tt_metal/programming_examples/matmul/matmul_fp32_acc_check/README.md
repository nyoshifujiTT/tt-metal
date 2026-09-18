# matmul_fp32_acc_check

Demonstrates that on Blackhole the accuracy of a matmul is limited by the intra-SOP alignment window inside a single `MVMUL`, not by `fp32_dest_acc_en`.

## Background

Each `MVMUL` reduces 16 K-elements as two independent 8-lane sum-of-products (SOP) groups. Within a group, every product is right-shifted to the group's maximum exponent. A product carries only ~12 bits (a 5-bit SrcA slice times a 7-bit SrcB slice), and there are no guard or sticky bits, so a term more than ~11 binades below its group maximum is dropped entirely. This happens *before* the FP32 accumulator, so `fp32_dest_acc_en=true` cannot recover it.

Accumulation across SOP groups, and across `MVMUL` instructions, is done at FP32 width and is not the limiting factor.

## What the example measures

The same 32 products (exponents spanning ~21 bits) are fed through two K-layouts with identical `fp32_dest_acc_en=true` and `MathFidelity::HiFi4`:

- `packed`: `K=32`, eight values per 8-lane SOP group, exponents spread across the group.
- `spread`: `K=256`, values placed every 8th slot so each SOP group holds at most one value.
- `uniform`: `K=32`, every lane of every SOP group carries a useful value, but all values within a group share one exponent.

Only the layout differs, so any accuracy difference is attributable to intra-SOP alignment. The `uniform` case is the control: a fully populated 8-wide group is not itself a problem, so the loss comes from the alignment shift rather than from the group width.

Inputs are tilized on the host. That is appropriate here because this is a test harness: on the
device, data stays in 32x32 tile layout end to end (matmul rejects non-tiled inputs and emits
tiled output), so a production path has no layout conversion at this point to begin with.

## Build

```bash
cd /home/ubuntu/tt-metal
./build_metal.sh --build-programming-examples --without-python-bindings --toolchain-path cmake/x86_64-linux-gcc-12-toolchain.cmake
```

## Run on ttsim

`soc_descriptor.yaml` must be in the same directory as `libttsim.so`.

```bash
cp tt_metal/soc_descriptors/blackhole_140_arch.yaml /home/ubuntu/ttsim/src/_out/release_bh/soc_descriptor.yaml
TT_METAL_SIMULATOR=/home/ubuntu/ttsim/src/_out/release_bh/libttsim.so \
  ./build_Release/programming_examples/metal_example_matmul_fp32_acc_check
```

## Run on real device

```bash
unset TT_METAL_SIMULATOR
./build_Release/programming_examples/metal_example_matmul_fp32_acc_check
```

## Inputs and pass criterion

Both operands vary along every axis, so a transposed or face-swapped operand cannot slip through
unnoticed:

```
A[m][k]  = (1 + m/M) * terms[k]        B[k][n] = (1 + n/N)
exact C[m][n] = (1 + m/M) * (1 + n/N) * sum_k terms[k]
```

`terms[k]` carries the exponent spread under test; the `m` and `n` factors make every output
element a distinct value. All 1024 elements are checked, not just element 0.

The reference is computed from the BF16 datums the device actually receives, not from the
pre-rounding floats. Comparing against the floats would fold BF16 input quantization into what is
meant to be a measurement of accumulator behaviour.

The pass criterion is Higham's forward error bound for an inner product (Accuracy and Stability of
Numerical Algorithms, 2nd ed., section 3.1):

```
|computed - exact| <= gamma_n * sum_k |a_k * b_k|,   gamma_n = n*u / (1 - n*u),   u = 2^-24
```

This is the bound a correct FP32-accumulating dot product must respect. `spread` and `uniform`
must satisfy it; `packed` must violate it. Earlier versions of this example used hand-picked
"effective bits" thresholds with no derivation behind them.

## B1: custom compute kernel

`kernels/compute/mm_custom.cpp` performs the same matmul without calling the LLK matmul library.
Neither `llk_unpack_AB_matmul*` nor `llk_math_matmul*` appears in it:

- the unpack side programs `Haloize_mode`, the ADC counters and the SrcA/SrcB datum counts, then
  issues the two `UNPACR`s itself, after taking the unpack context and posting the semaphore;
- the math side programs `ADDR_MOD_0/1/2/4/5` and issues the 16-`MVMUL` full-tile sequence
  directly, once per fidelity phase, with no MOP and no replay buffer.

This is a prerequisite for the planned zero-injection work, which has to change what lands in
SrcA/SrcB between the unpack and the MVMULs and has to control the MVMUL issue order. Neither is
reachable through `matmul_tiles()` or through the LLK matmul entry points, which hide both behind
a MOP.

Restrictions: full 32x32 tiles (4 faces per operand), no transpose, `ct_dim = rt_dim = 1`.

Two details cost real debugging time and are worth recording. The closing `SETRWC` must release
**both** source registers: `CLR_A` after the last fidelity phase and `CLR_B` at the end of the
reuse row. Releasing only SrcA leaves SrcB permanently valid and the unpacker blocks forever on
the next tile. Separately, `TTI_*` macros expand to inline asm statements, so they cannot be
placed inside `UNPACK((...))` / `MATH((...))`, which require an expression; they have to live in a
function that the macro then calls.

The example runs both kernels on all three layouts and requires bit-exact agreement. The check was
validated with a negative control: changing the addr_mod of the final MVMUL in the custom kernel
turns all three checks into `NG` (128/1024 elements differing).

## B2: the same matmul with M contiguous

`kernels/compute/mm_custom_b2.cpp` answers a question B1 raises. B1 issues the 16 MVMULs in the
order the LLK uses, which splits M into a face index (outer) and an 8-row half (inner) with N in
between:

```
B0A0 B0A0 B0A1 B0A1  B2A0 B2A0 B2A1 B2A1  B1A2 B1A2 B1A3 B1A3  B3A2 B3A2 B3A3 B3A3
```

That split is inherited, not required. It dates to Grayskull, where the MOP was a fixed two-level
loop: with the ColMajor dest face layout 16 MVMULs fit one MOP run, while RowMajor only fit 8 and
needed extra `SETRWC`s in the math thread. The Grayskull code shows this directly, as
`ckernel_template tmp(2, 8, ...)` for ColMajor against `tmp(2, 4, ...)` for RowMajor. Blackhole has
REPLAY and does not have that constraint, ColMajor was deleted for Wormhole B0 and the
`DstTileFaceLayout` parameter was removed entirely, yet the instruction order carried over. See
tt-metal#3546 and tt-metal#5420 for the contemporaneous discussion.

Nothing else was found to depend on the order. The packer can start anywhere in Dst and can pack
row-granular (`pack_rows`, 1 to 64 rows), and the math/pack handshake is per Dst section rather
than per row, so neither constrains how math fills a tile.

B2 therefore walks `for k in 0,1: for j in 0,1: for i in 0..3` with M contiguous and innermost.
Same 16 products, same Dst, different issue order, so the accumulation order differs and the
result is *not* bit-identical to B1. It is judged against the same Higham bound instead.

Measured, identical on ttsim and on Blackhole p150b silicon, and identical to B1's ratios:

```
check=b2_no_i_split_spread  expected=within_bound actual=within_bound result=OK
detail b2 spread  err_over_bound=0.0247595  elements_within=1024/1024
check=b2_no_i_split_uniform expected=within_bound actual=within_bound result=OK
detail b2 uniform err_over_bound=0          elements_within=1024/1024
check=b2_no_i_split_packed  expected=over_bound   actual=over_bound   result=OK
detail b2 packed  err_over_bound=620.646    elements_within=12/1024
```

So the split is safe to drop: M can be kept contiguous with the same four increments, the same
instruction count, and the same accuracy.

One detail cost real debugging time. Moving Dst backwards cannot be done by wrapping. The SrcA and
SrcB counters are 6-bit and wrap at 64 rows, but the Dst counter is 10-bit (`uint10_t Dst, Dst_Cr`
in the ISA's RWCs) and wraps at 1024, so a `dest` increment of 40 intended as -24 just keeps
climbing: measured, Dst went 0,8,32,40 then 80,88,112,120,128 and never came back. The backward
moves use the `cr` marker instead - advance the marker by 16 on the `j` step, clear it on the `k`
step.

## How to read output

- `check=*_layout_worst_element`: expected vs actual at the element with the largest bound-relative
  error
- `check=*_layout_err_over_fp32_bound`: worst `|error| / bound`. At or below 1 means the layout
  behaves like a correct FP32 accumulation
- `detail elements_within_fp32_bound`: how many of the 1024 outputs satisfy the bound
- `check=custom_kernel_bitexact_*`: number of elements where the B1 kernel differs from the
  compute-API kernel; must be 0
- `note ... mvmul_instruction_ratio=`: instruction-count cost of the spread layout

Measured on Blackhole p150b silicon:

```
note gamma_32=1.907352270798246e-06 gamma_256=1.5259021896696422e-05 checked_elements=1024
check=spread_layout_err_over_fp32_bound expected=1 actual=0.0247595 result=OK
check=uniform_layout_err_over_fp32_bound expected=1 actual=0 result=OK
check=packed_layout_err_over_fp32_bound expected=1 actual=620.646 result=OK
detail elements_within_fp32_bound spread=1024/1024 uniform=1024/1024 packed=12/1024
detail packed worst_index=162 expected=5.714314600452781 actual=5.707550048828125 ratio=620.646
check=custom_kernel_bitexact_packed expected=0 actual=0 result=OK
```

`spread` and `uniform` satisfy the FP32 bound on every element. `packed` exceeds it by a factor of
620 and satisfies it on only 12 of 1024 elements, despite identical `fp32_dest_acc_en=true` and
`MathFidelity::HiFi4`. Only the K-layout differs.

The practical consequence is that full FP32-class accuracy currently requires each SOP group to
span at most ~11 binades. Data that already satisfies this keeps FP32-class accuracy at full
multiplier utilisation, as the `uniform` case shows. Data that does not must be spread to one
useful value per SOP group, which costs a factor of 8 in multiplier utilisation.

## C: zero injection

The `packed` layout above loses 12 bits because eight values with different exponents share one
SOP group. `kernels/compute/mm_zero_inject.cpp` runs that same layout with at most one useful
value per group, and the result satisfies the FP32 bound.

Measured on ttsim (Blackhole), same input, same `fp32_dest_acc_en=true`, same `MathFidelity::HiFi4`
as the `packed` run:

```
check=zero_inject_err_over_fp32_bound expected=1 actual=0.265 result=OK
detail zero_inject elements_within_fp32_bound=1024/1024
detail zero_inject_vs_baseline packed_ratio=620.646 zero_inject_ratio=0.265
```

620.6 to 0.265, on all 1024 elements. Only the SrcA occupancy differs.

### Only SrcA is zeroed

`mvmul()` tests the operands with an OR:

```c
zero_a = (a & 0x3FFFF) < 0x400;  zero_b = ...;
if (zero_a || zero_b) { signs[i] = 0; exps[i] = zero_term_exp; mans[i] = 0; }
```

A zero in SrcA alone kills the lane, and the lane's exponent is forced to a neutral value instead
of anything derived from the data, so it cannot raise the group maximum that the other lanes are
aligned to. SrcB in those lanes is never read, so it needs no zero padding and no reordering.

### Placement

SOP groups are cut at lanes 0-7 and 8-15, and lane `i` reads SrcA row `src_a_row + i` and SrcB
column `i`. The two constraints together fix the layout: a K value must sit at the same index on
both sides, so SrcA row `r` must hold K=`r`, and the two useful values must be 8 rows apart to
fall in different groups. Each step therefore takes K=`r` and K=`r+8` of one 16-wide K window,
and 16 steps cover the 32 K values of a tile.

SrcB is not moved to select the K window, because `SETRWC` cannot place its row counter at 16
(`rwc_b` accepts only 0 or 8). Instead the unpacker re-points SrcB's L1 base at the face pair for
the current K half, which it can do freely since SrcB is re-fetched every step anyway.

### Maintaining the zeros

`ZEROSRC` clears whole banks, so it cannot maintain the invariant incrementally. Clearing
individual rows by unpacking from L1's zero region also failed: the clearing `UNPACR`s advance the
config context themselves, so a clear cannot be aimed reliably at the bank holding the rows it was
meant to undo. Measured attempts left the window accumulating stale rows (0,8 then 0,2,8,10 then
0,2,4,6,8,10,12,14 when clearing one pair, and an interleaved mess when clearing the whole
history), and each stale row contributes a K value that does not belong to the step.

What works is `UNPACR_NOP` with the stall-and-clear encoding at the top of each step: it zeroes the
unpacker-side bank and waits until the matrix unit has released it. That makes the invariant local
- on entry to the row writes, every row of the bank is zero - at one extra instruction per step.

### Config contexts

Every cfg register this kernel writes has to be written in both context slots, because the kernel
alternates config contexts per step and the unpacker reads the slot for the current one. This
applies to the SrcA L1 base, the SrcA Dest address, and the SrcB L1 base. Writing only the
context-0 slot is not a small error: it left every context-1 step reading a stale address, which
showed up as exactly half the output tile being wrong.

### MVMUL sequence

Both operands are four stacked 16x16 faces (TL, TR, BL, BR). SrcB is in0, M-by-K, so its faces
are M0-15/K0-15, M0-15/K16-31, M16-31/K0-15, M16-31/K16-31. SrcA is in1, K-by-N, and this kernel
writes the K rows of both N faces: SrcA rows 0-15 feed output columns 0-15, rows 16-31 feed
columns 16-31.

A step is 8 MVMULs per fidelity phase: {SrcB M0-15, M16-31} x {SrcA N0-15, N16-31}, each covering
8 of the 64 Dst rows an fp32 32x32 tile occupies. Note that Dst increments are not modular, so
returning the Dst counter to row 16 for the second SrcA face is done with its `cr` marker, not by
adding 40.

### Cost

16 steps per tile instead of 2 MVMUL groups, i.e. the 8x MVMUL increase the analysis predicted.
The unpacker writes one row at a time instead of a whole tile, which does not reduce the bytes
moved, only splits them, and adds one bank-clear per step.

### Verifying changes to this kernel

Every failure above was diagnosed by instrumenting ttsim rather than by inspection: printing the
non-zero SrcA rows per MVMUL, the SrcA write row and source address, and the SrcB base address per
context. The failure modes look alike from the outside - a wrong result or a zero result - so if
this kernel is modified, instrument rather than guess.
