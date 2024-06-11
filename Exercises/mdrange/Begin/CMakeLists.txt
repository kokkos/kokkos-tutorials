cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialMdRange)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

add_executable(mdrange_exercise exercise_mdrange_begin.cpp)
target_link_libraries(mdrange_exercise Kokkos::kokkos)
