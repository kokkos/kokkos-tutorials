cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialTeamVectorLoop)
include(../../common.cmake)

add_executable(TeamVectorLoop team_vector_loop_begin.cpp)
target_link_libraries(TeamVectorLoop Kokkos::kokkos)


