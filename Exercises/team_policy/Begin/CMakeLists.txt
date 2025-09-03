cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialTeamPolicy)

# Add a custom module path for find_package(Kokkos)
include(../../SetUpKokkos.cmake)

add_executable(TeamPolicy team_policy_begin.cpp)
target_link_libraries(TeamPolicy Kokkos::kokkos)
