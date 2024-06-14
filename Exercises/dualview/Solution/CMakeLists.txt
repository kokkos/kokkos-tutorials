cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialDualView)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

add_executable(dualview_Solution dual_view_exercise.cpp)
target_link_libraries(dualview_Solution Kokkos::kokkos)
