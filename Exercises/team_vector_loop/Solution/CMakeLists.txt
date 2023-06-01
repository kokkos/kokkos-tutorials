cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTeamVectorLoop)
include(../../common.cmake)

add_executable(TeamVectorLoop team_vector_loop_solution.cpp)
target_link_libraries(TeamVectorLoop Kokkos::kokkos)


