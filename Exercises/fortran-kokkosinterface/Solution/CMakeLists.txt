cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialFortran)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

enable_language(Fortran)

add_executable(ftest_solution abi.f90 f_interface.f90 main.f90 c_interface.cpp)
target_link_libraries(ftest_solution Kokkos::kokkos)
