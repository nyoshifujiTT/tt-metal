add_executable(metal_example_matmul_fp32_acc_check)
target_sources(metal_example_matmul_fp32_acc_check PRIVATE matmul_fp32_acc_check.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(
    metal_example_matmul_fp32_acc_check
    PRIVATE
        TT::Metalium
        Matmul::Common
        MPI::MPI_CXX
)
