cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTeamPolicy)
include(../../common.cmake)

add_executable(TeamPolicy_Solution team_policy_solution.cpp)
target_link_libraries(TeamPolicy_Solution Kokkos::kokkos)
