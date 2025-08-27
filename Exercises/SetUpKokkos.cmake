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

set(KokkosTutorials_KOKKOS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/dep/kokkos" CACHE PATH "Where Kokkos sources are located")

find_package(Kokkos QUIET)

if (Kokkos_FOUND)
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else ()
    if (EXISTS ${KokkosTutorials_KOKKOS_SOURCE_DIR})
        message(STATUS "Using Kokkos from ${KokkosTutorials_KOKKOS_SOURCE_DIR}")
        add_subdirectory(${KokkosTutorials_KOKKOS_SOURCE_DIR} Kokkos)
    else ()
        message(STATUS "Downloading Kokkos to ${KokkosTutorials_KOKKOS_SOURCE_DIR}")
        include(FetchContent)
        FetchContent_Declare(
                Kokkos
                URL      https://github.com/kokkos/kokkos/releases/download/4.6.02/kokkos-4.6.02.tar.gz
                URL_HASH SHA256=baf1ebbe67abe2bbb8bb6aed81b4247d53ae98ab8475e516d9c87e87fa2422ce
                SOURCE_DIR ${KokkosTutorials_KOKKOS_SOURCE_DIR}
                DOWNLOAD_EXTRACT_TIMESTAMP ON
        )
        FetchContent_MakeAvailable(Kokkos)
    endif ()
endif ()
