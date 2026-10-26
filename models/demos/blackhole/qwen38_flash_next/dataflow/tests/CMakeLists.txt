# Device tests of the Qwen3.8-Flash-Next BH-GLX dataflow implementation (README section 12: the chosen decode form
# DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 tiles | DCP1] at ctx 1, B=1, k=1). One gtest binary per level:
#   test_qwen38_dataflow_probe    the machine: the 8 x 4 torus as the Ethernet links see it, per chip and neighbour
#   test_qwen38_dataflow_meeting  I-1b: a group's all-gather sent directly from the Ethernet cores along the BFS trees
#   test_qwen38_dataflow_link     I-1d: the sustained rate of one and of both channels of a link (channels per direction)
# Built with the tt_metal tests (TT_METAL_BUILD_TESTS); executables in build/test/models/demos/blackhole/qwen38_flash_next/.

set(QWEN38_DATAFLOW_INCLUDE_DIRS
    "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
    ${PROJECT_SOURCE_DIR}/tests
    ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
    ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

foreach(GTEST_TARGET test_qwen38_dataflow_probe test_qwen38_dataflow_meeting test_qwen38_dataflow_link)
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
