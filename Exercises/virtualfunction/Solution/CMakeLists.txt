cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialVirtualFunction)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

add_executable(virtual_function_solution virtual_function.cpp classes.cpp)
target_link_libraries(virtual_function_solution Kokkos::kokkos)
