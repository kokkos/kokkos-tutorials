cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTeamVectorLoop)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

add_executable(TeamVectorLoop_Solution team_vector_loop_solution.cpp)
target_link_libraries(TeamVectorLoop_Solution Kokkos::kokkos)
