# Convenience macro to warn the user if a GPU backend is enabled
macro(KokkosTutorials_WarnGPU)
    if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL OR Kokkos_ENABLE_OPENMPTARGET OR Kokkos_ENABLE_HPX)
        message(WARNING "cmake"
                "a Kokkos accelerator backend is enabled, it might cause issue with the current program"
                "Please recompile with only a host backend enabled (e.g. -DKokkos_ENABLE_OPENMP=ON)")
    endif ()
endmacro()

# Early return if Kokkos is already set up
# We do not use Kokkos_FOUND as it is not globally defined
if (TARGET Kokkos::kokkos)
    return()
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(default_build_type "RelWithDebInfo")
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING
            "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
            FORCE)
endif ()

# Where to find Kokkos' source code. This might be set by the user.
# In order to automatically share the download between exercises when they are compiled individually,
# the default directory is inside the source tree.
# This might break if the default in source directory is called from multiple cmake instances at the same time.

set(KokkosTutorials_KOKKOS_SOURCE_DIR "dep/kokkos" CACHE PATH "Description for KokkosTutorials_KOKKOS_SOURCE_DIR")

find_package(Kokkos CONFIG)

if (Kokkos_FOUND)
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else ()
    if (EXISTS ${KokkosTutorials_KOKKOS_SOURCE_DIR})
        add_subdirectory(${KokkosTutorials_KOKKOS_SOURCE_DIR} Kokkos)
    else ()
        cmake_policy(VERSION 3.24)  # Use extract timestamp for fetch content
        include(FetchContent)
        FetchContent_Declare(
                Kokkos
                URL      https://github.com/kokkos/kokkos/releases/download/4.5.01/kokkos-4.5.01.tar.gz
                URL_HASH SHA256=52d003ffbbe05f30c89966e4009c017efb1662b02b2b73190670d3418719564c
                SOURCE_DIR ${KokkosTutorials_KOKKOS_SOURCE_DIR}
        )
        FetchContent_MakeAvailable(Kokkos)
    endif ()
endif ()
