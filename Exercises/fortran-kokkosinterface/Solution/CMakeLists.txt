cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialFortran)

# Add a custom module path for find_package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../.cmake")

find_package(Kokkos REQUIRED)

execute_process (
    COMMAND bash -c "printenv"
    OUTPUT_VARIABLE outVar
)
message("${outVar}")

enable_language(Fortran)

add_executable(ftest_solution abi.f90 f_interface.f90 main.f90 c_interface.cpp)
target_link_libraries(ftest_solution Kokkos::kokkos)
