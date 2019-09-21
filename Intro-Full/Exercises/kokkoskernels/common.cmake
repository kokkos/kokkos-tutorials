
set(SPACK_CXX $ENV{SPACK_CXX})
if(SPACK_CXX)
  message("found spack compiler ${SPACK_CXX}")
  set(CMAKE_CXX_COMPILER ${SPACK_CXX} CACHE STRING "the C++ compiler" FORCE)  
  set(ENV{CXX} ${SPACK_CXX})
endif()

cmake_policy(SET CMP0074 NEW)
find_package(KokkosKernels REQUIRED)
