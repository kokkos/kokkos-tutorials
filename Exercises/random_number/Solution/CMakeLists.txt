cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialRNG)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(random_number_solution MC_DartSampler.cpp)
target_link_libraries(random_number_solution Kokkos::kokkos)
