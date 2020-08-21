cmake_minimum_required (VERSION 3.12)
project (KokkosTutorialVectorShift)

find_package(Kokkos REQUIRED)

add_executable(Vectorshift vectorshift_solution.cpp)
target_link_libraries(Vectorshift Kokkos::kokkos)

find_package(MPI REQUIRED)
target_link_libraries(Vectorshift MPI::MPI_CXX)
