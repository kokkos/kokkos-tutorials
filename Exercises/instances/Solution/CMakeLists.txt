cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialInstances)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

add_executable(Instances_solution instances_solution.cpp)
target_link_libraries(Instances_solution Kokkos::kokkos)
