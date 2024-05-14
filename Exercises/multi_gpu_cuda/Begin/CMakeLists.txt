cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialMultiGpuCuda)
include(../../common.cmake)
  
add_executable(ExerciseMultiGPU multi_gpu_cuda.cpp)
target_link_libraries(ExerciseMultiGPU Kokkos::kokkos)

