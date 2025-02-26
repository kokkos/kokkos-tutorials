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

set(KokkosTutorials_KOKKOS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../dep/kokkos" CACHE PATH "Description for KokkosTutorials_KOKKOS_SOURCE_DIR")

if (NOT KokkosTutorials_FORCE_INTERNAL_Kokkos)
    find_package(Kokkos CONFIG)
endif ()

if (Kokkos_FOUND)
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
elseif (NOT KokkosTutorials_FORCE_EXTERNAL_Kokkos)
    if (EXISTS ${KokkosTutorials_KOKKOS_SOURCE_DIR})
        add_subdirectory(${KokkosTutorials_KOKKOS_SOURCE_DIR} Kokkos)
    else ()
        include(FetchContent)
        FetchContent_Declare(
                Kokkos
                GIT_REPOSITORY https://github.com/kokkos/kokkos.git
                GIT_TAG 4.5.01
                SOURCE_DIR ${KokkosTutorials_KOKKOS_SOURCE_DIR}
        )
        FetchContent_MakeAvailable(Kokkos)
        set(Kokkos_FOUND True)
    endif ()
endif ()
