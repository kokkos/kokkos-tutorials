#!/bin/bash -ex
cmake3 $HOME/Kokkos/kokkos \
-DCMAKE_INSTALL_PREFIX=$HOME/Kokkos/kokkos-cmake-install \
-DKokkos_ENABLE_CUDA=ON \
-DKokkos_ENABLE_SERIAL=ON \
-DCMAKE_CXX_COMPILER=$HOME/Kokkos/kokkos/bin/nvcc_wrapper \
-DKokkos_ARCH_BDW=ON \
-DKokkos_ARCH_VOLTA70=ON \
-DKokkos_ENABLE_DEPRECATED_CODE=OFF \
-DKokkos_ENABLE_CUDA_LAMBDA=ON \
-DCMAKE_CXX_FLAGS="-O3 -g"
