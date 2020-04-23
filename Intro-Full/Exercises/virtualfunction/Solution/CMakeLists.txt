cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialVirtualFunction)
include(../../common.cmake)

add_executable(virtual_function virtual_function.cpp classes.cpp)
target_link_libraries(virtual_function Kokkos::kokkos)
target_include_directories(virtual_function PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


