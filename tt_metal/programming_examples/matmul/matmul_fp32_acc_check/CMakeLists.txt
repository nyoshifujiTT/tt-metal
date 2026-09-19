add_executable(metal_example_matmul_fp32_acc_check)
target_sources(metal_example_matmul_fp32_acc_check PRIVATE matmul_fp32_acc_check.cpp)

add_executable(metal_example_matmul_fp32_accurate)
target_sources(metal_example_matmul_fp32_accurate PRIVATE matmul_fp32_accurate.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
find_package(MPI REQUIRED COMPONENTS CXX)

foreach(_tgt metal_example_matmul_fp32_acc_check metal_example_matmul_fp32_accurate)
    target_link_libraries(
        ${_tgt}
        PRIVATE
            TT::Metalium
            Matmul::Common
            MPI::MPI_CXX
    )
endforeach()
