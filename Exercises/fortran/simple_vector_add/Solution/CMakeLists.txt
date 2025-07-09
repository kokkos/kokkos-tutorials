cmake_minimum_required(VERSION 3.16)
project(KokkosFortranTutotial01 LANGUAGES Fortran CXX)
include(../../../common.cmake)

find_package(flcl)

add_executable(exercise exercise.f90 axpy-view-f.f90 axpy-view-cxx.cpp)
target_link_libraries(exercise flcl::flcl)
