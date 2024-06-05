cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialVirtualFunction)

# Add a custom module path for find_package
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake)

find_package(Kokkos REQUIRED)

add_executable(virtual_function_solution virtual_function.cpp classes.cpp)
target_link_libraries(virtual_function_solution Kokkos::kokkos)
