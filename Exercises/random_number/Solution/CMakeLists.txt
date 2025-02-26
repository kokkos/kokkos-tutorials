cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialRNG)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

find_package(Kokkos REQUIRED)

add_executable(random_number_solution MC_DartSampler.cpp)
target_link_libraries(random_number_solution Kokkos::kokkos)
