# matmul_fp32_acc_check

This is a single-core matmul (`32x32x32`) precision check example using `fp32_dest_acc_en`.

- `K=32` (=`Kt=1`), so `matmul_tiles` runs exactly once.
- Inputs are BF16 and outputs are BF16.
- It compares `fp32_dest_acc_en=true` and `false` on identical inputs, and checks which one is closer to the FP32 reference.

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

The output begins with `input_terms=[...]`.
These 32 values are the K-dimension accumulation terms (since every `A` element is `1.0`, each output element is their sum).

Key lines:

- `check=reference_fp32_value ...`: FP32 reference value (sum in FP32)
- `check=reference_fp32_rounded_to_bf16 ...`: FP32 reference value rounded once to BF16 at the end
- `check=actual_fp32_mode ...`: measured output with `fp32_dest_acc_en=true`
- `check=actual_non_fp32_mode ...`: measured output with `fp32_dest_acc_en=false`
- `check=actual_fp32_mode_is_closer_to_reference_fp32_than_non_fp32_mode ...`: whether `true` is closer to the FP32 reference
- `check=overall ... result=OK`: final pass/fail result

Example on `ttsim`:

- `reference_fp32_value = -0.4453125`
- `actual_fp32_mode = -0.4453125`
- `actual_non_fp32_mode = -0.44921875`

In this test case, only `fp32_dest_acc_en=true` matches the FP32 reference.
