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

## How to read output

Key lines:

- `check=packed_layout_effective_bits ...`: effective precision when values with differing exponents share SOP groups
- `check=spread_layout_effective_bits ...`: effective precision with one value per SOP group
- `check=uniform_exponent_layout_effective_bits ...`: effective precision with a full 8-wide group whose exponents agree
- `check=layout_alone_changes_accuracy ...`: confirms the layout is the only variable
- `note ... mvmul_instruction_ratio=...`: instruction-count cost of the spread layout

Measured on `ttsim` (Blackhole):

- `packed` (`K=32`): 12.7 effective bits
- `spread` (`K=256`): 24.2 effective bits, exact to the FP32 reference
- `uniform` (`K=32`): 24.0 effective bits, exact to the FP32 reference, at full multiplier utilisation
- Cost: `MVMUL` instruction count scales with `Kt`, so the spread layout needs 8x the instructions for the same number of useful products (measured: 222 instructions at `Kt=1` versus 947 at `Kt=8`, i.e. ~104 per `matmul_tiles` plus fixed overhead).

The practical consequence is that full FP32-class accuracy currently requires each SOP group to span at most ~11 binades. Data that already satisfies this keeps FP32-class accuracy at full multiplier utilisation, as the `uniform` case shows. Data that does not must be spread to one useful value per SOP group, which costs a factor of 8 in multiplier utilisation.
