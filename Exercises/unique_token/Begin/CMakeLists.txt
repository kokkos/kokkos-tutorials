cmake_minimum_required (VERSION 3.10)
project (KokkosTutorialUniqueToken)
include(../../common.cmake)

add_executable(uniquetoken unique_token.cpp)
target_link_libraries(uniquetoken Kokkos::kokkos)


