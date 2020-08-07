cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialTeamPolicy)
include(../../common.cmake)

add_executable(TeamPolicy team_policy_begin.cpp)
target_link_libraries(TeamPolicy Kokkos::kokkos)


