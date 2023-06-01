
set(SPACK_CXX $ENV{SPACK_CXX})
if(SPACK_CXX)
  message("found spack compiler ${SPACK_CXX}")
  set(CMAKE_CXX_COMPILER ${SPACK_CXX} CACHE STRING "the C++ compiler" FORCE)
  set(ENV{CXX} ${SPACK_CXX})
endif()

include(FetchContent)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  # try find_package first before trying to download Kokkos
  set(FETCH_CONTENT_EXTRA_ARGS FIND_PACKAGE_ARGS NAMES Kokkos)
endif()
FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG        4.0.01
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../dep/Kokkos
  ${FETCH_CONTENT_EXTRA_ARGS}
)
FetchContent_MakeAvailable(Kokkos)

find_package(Kokkos REQUIRED)
