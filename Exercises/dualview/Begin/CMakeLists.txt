cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialDualView)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

add_executable(dualview dual_view_exercise.cpp)
target_link_libraries(dualview Kokkos::kokkos)
