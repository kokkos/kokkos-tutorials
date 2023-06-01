cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTeamPolicy)
include(../../common.cmake)

add_executable(TeamPolicy team_policy_solution.cpp)
target_link_libraries(TeamPolicy Kokkos::kokkos)


