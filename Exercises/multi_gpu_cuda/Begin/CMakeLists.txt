cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialMultiGpuCuda)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

if (NOT Kokkos_ENABLE_CUDA)
    message(WARNING "This example requires CUDA, enable with -DKokkos_ENABLE_CUDA=ON")
    return()
endif()

add_executable(MultiGPU_exercise multi_gpu_cuda.cpp)
target_link_libraries(MultiGPU_exercise Kokkos::kokkos)
