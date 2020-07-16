cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialMdRange)
include(../../common.cmake)

add_executable(mdrange_exercise exercise_mdrange_solution.cpp)
target_link_libraries(mdrange_exercise Kokkos::kokkos)


