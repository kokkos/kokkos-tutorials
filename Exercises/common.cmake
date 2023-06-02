
set(SPACK_CXX $ENV{SPACK_CXX})
if(SPACK_CXX)
  message("found spack compiler ${SPACK_CXX}")
  set(CMAKE_CXX_COMPILER ${SPACK_CXX} CACHE STRING "the C++ compiler" FORCE)
  set(ENV{CXX} ${SPACK_CXX})
endif()

set(Kokkos_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../dep/Kokkos)

include(FetchContent)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        4.0.01
    SOURCE_DIR ${Kokkos_COMMON_SOURCE_DIR}
    FIND_PACKAGE_ARGS
  )
  FetchContent_MakeAvailable(Kokkos)
  
  find_package(Kokkos REQUIRED)
else()
  find_package(Kokkos)
  if(NOT Kokkos_FOUND)
    FetchContent_Declare(
      Kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG        4.0.01
      SOURCE_DIR ${Kokkos_COMMON_SOURCE_DIR}
    )
    FetchContent_MakeAvailable(Kokkos)
  endif()
endif()
