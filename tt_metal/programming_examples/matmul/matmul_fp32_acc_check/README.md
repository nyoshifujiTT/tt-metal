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
