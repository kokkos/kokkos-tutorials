cmake_minimum_required (VERSION 3.10)
project (KokkosKernelsTeamGemm)
include(../../common.cmake)

add_executable(teamgemm teamgemm.cpp)
target_link_libraries(teamgemm Kokkos::kokkoskernels)


