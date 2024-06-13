cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialFortran)
include(../../common.cmake)

execute_process (
    COMMAND bash -c "printenv"
    OUTPUT_VARIABLE outVar
)
message("${outVar}")

enable_language(Fortran)

add_executable(ftest.x abi.f90 f_interface.f90 main.f90 c_interface.cpp)
target_link_libraries(ftest.x Kokkos::kokkos)
