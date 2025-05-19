cmake_minimum_required(VERSION 3.19)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(exercise LANGUAGES Fortran C CXX)

find_package(Kokkos)
find_package(flcl)

add_executable(exercise exercise.f90 axpy-view-f.f90 axpy-view-cxx.cc)
set_target_properties(
    exercise
    PROPERTIES
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod"
)
target_include_directories(exercise
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>
        $<INSTALL_INTERFACE:mod>
)
target_include_directories(exercise
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(exercise ${PROJECT_LIBS} flcl::flcl)
set_target_properties(exercise PROPERTIES LINKER_LANGUAGE Fortran)
