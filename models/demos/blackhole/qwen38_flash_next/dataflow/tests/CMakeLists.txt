# Device tests of the Qwen3.8-Flash-Next BH-GLX dataflow implementation (README section 12: the chosen decode form
# DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 tiles | DCP1] at ctx 1, B=1, k=1). One gtest binary per level:
#   test_qwen38_dataflow_probe    the machine: the 8 x 4 torus as the Ethernet links see it, per chip and neighbour
#   test_qwen38_dataflow_meeting  I-1b: a group's all-gather sent directly from the Ethernet cores along the BFS trees
#   test_qwen38_dataflow_meeting2 I-1b': the meeting in two planes (channel parity) with receive rings and staging rings
#   test_qwen38_dataflow_meeting3 I-1b'': version 2 with frames never rewritten after the root (per-frame tags)
#   test_qwen38_dataflow_meeting4 I-1b''': version 3 without flow control (rings sized by the causality bound)
#   test_qwen38_dataflow_meeting5 I-1b v5: every group collective (AG, RS direct) on one table-driven kernel
#   test_qwen38_dataflow_link     I-1d: the sustained rate of one and of both channels of a link (channels per direction)
#   test_qwen38_dataflow_costs    I-1e: the cycle costs of the relay steps on an Ethernet core RISC
#   test_qwen38_dataflow_mm       I-1c: the Tensix matmul unit cost (HiFi2, bf8 weights, fp32 DEST) on one core
#   test_qwen38_dataflow_mm_chain I-1c-2: one chip's in_proj pool on 130 cores (Stream-K, fp32 partials over the NoC)
#   test_qwen38_dataflow_dram     I-1b': the DRAM lane, the cores reading their own slices into their L1 at once
# Built with the tt_metal tests (TT_METAL_BUILD_TESTS); executables in build/test/models/demos/blackhole/qwen38_flash_next/.

set(QWEN38_DATAFLOW_INCLUDE_DIRS
    "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
    ${PROJECT_SOURCE_DIR}/tests
    ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
    ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

foreach(GTEST_TARGET test_qwen38_dataflow_probe test_qwen38_dataflow_meeting test_qwen38_dataflow_meeting2 test_qwen38_dataflow_meeting3 test_qwen38_dataflow_meeting4 test_qwen38_dataflow_meeting5 test_qwen38_dataflow_link test_qwen38_dataflow_costs test_qwen38_dataflow_mm test_qwen38_dataflow_mm_chain test_qwen38_dataflow_dram)
    add_executable(${GTEST_TARGET} ${GTEST_TARGET}.cpp)
    target_precompile_headers(${GTEST_TARGET} REUSE_FROM metal_common_pch)
    target_link_libraries(${GTEST_TARGET} PRIVATE test_metal_common_libs)
    target_include_directories(${GTEST_TARGET} BEFORE PRIVATE ${QWEN38_DATAFLOW_INCLUDE_DIRS})
    set_target_properties(
        ${GTEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/models/demos/blackhole/qwen38_flash_next
    )
    target_compile_options(${GTEST_TARGET} PUBLIC ${COMPILE_OPTIONS})
    list(APPEND QWEN38_DATAFLOW_TEST_TARGETS ${GTEST_TARGET})
endforeach()

add_custom_target(qwen38_dataflow_tests DEPENDS ${QWEN38_DATAFLOW_TEST_TARGETS})
