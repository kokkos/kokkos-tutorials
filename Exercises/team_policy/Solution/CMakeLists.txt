cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTeamPolicy)

# Add a custom module path for find_package(Kokkos)
include(../../cmake/SetUpKokkos.cmake)

add_executable(TeamPolicy_Solution team_policy_solution.cpp)
target_link_libraries(TeamPolicy_Solution Kokkos::kokkos)
