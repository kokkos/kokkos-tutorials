#!/bin/bash

rm -r CMakeCache.txt CMakeFiles cmake_install.cmake LibInstalls

KOKKOS_PATH=${HOME}/Kokkos/kokkos
KOKKOSKERNELS_PATH=${HOME}/Kokkos/kokkos-kernels

CXX_FLAG=
KOKKOS_ARCH_FLAG=

KOKKOS_DEVICES="Cuda,OpenMP"
# Override the compiler and CPU/GPU architectures here:
# - add value after --compiler= or --arch=
# - uncomment the line
#CXX_FLAG="--compiler="
#KOKKOS_ARCH_FLAG="--arch="
TPLS=
OPTIONS=
CUDA_OPTIONS=

EXERCISE_DIR=${PWD}
mkdir -p LibInstalls
KOKKOSKERNELS_INSTALL=${EXERCISE_DIR}/LibInstalls/kernels-install
cd LibInstalls

CONFIG_CMD="${KOKKOSKERNELS_PATH}/cm_generate_makefile.bash $CXX_FLAG --with-devices=${KOKKOS_DEVICES} --kokkos-path=${KOKKOS_PATH} --kokkoskernels-path=${KOKKOSKERNELS_PATH} --prefix=${KOKKOSKERNELS_INSTALL} --with-options=${OPTIONS} --with-cuda-options=${CUDA_OPTIONS} $KOKKOS_ARCH_FLAG --with-tpls=${TPLS} --kokkos-make-j=8 --disable-kokkos-tests --disable-tests --disable-examples --no-default-eti"

echo $CONFIG_CMD
$CONFIG_CMD

make install -j8

cd ${EXERCISE_DIR}

echo "KERNELS_INSTALL_PATH = $KOKKOSKERNELS_INSTALL"
cmake -DKokkosKernels_ROOT="${KOKKOSKERNELS_INSTALL}" .
