mum_required (VERSION 3.12)
project (KokkosTutorialVectorShift)

find_package(KokkosRemote REQUIRED)

add_executable(Vectorshift vectorshift.cpp)
target_link_libraries(Vectorshift Kokkos::kokkosremote)

find_package(MPI REQUIRED)
target_link_libraries(Vectorshift MPI::MPI_CXX)
