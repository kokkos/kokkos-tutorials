#!/bin/bash
#export KOKKOS_ROOT_DIR=/KOKKOS/DIR/HERE
rm *.o *.mod *.x
gfortran -c -std=f2008 abi.f90
gfortran -c -std=f2008 f_interface.f90
g++ -c -fopenmp -I. -I$KOKKOS_ROOT_DIR/include c_interface.cpp
gfortran -c -g -std=f2008 main.f90
gfortran -std=f2008 -o ftest.x abi.o f_interface.o c_interface.o main.o -L$KOKKOS_ROOT_DIR/lib -lkokkos -lstdc++ -fopenmp
