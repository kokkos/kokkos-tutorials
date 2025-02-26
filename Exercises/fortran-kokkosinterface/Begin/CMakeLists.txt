cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialFortran)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

enable_language(Fortran)

add_executable(ftest_exercise abi.f90 f_interface.f90 main.f90 c_interface.cpp)
target_link_libraries(ftest_exercise Kokkos::kokkos)
