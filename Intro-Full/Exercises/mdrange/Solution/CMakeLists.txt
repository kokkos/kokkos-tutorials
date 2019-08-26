cmake_minimum_required (VERSION 3.10)
include(../../common.cmake)

project (KokkosTutorialMdRange)
add_executable(mdrange_exercise exercise_mdrange_solution.cpp)
target_link_libraries(mdrange_exercise Kokkos::kokkos)


