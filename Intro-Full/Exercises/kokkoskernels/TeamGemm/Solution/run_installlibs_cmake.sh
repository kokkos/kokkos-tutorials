rm -r CMakeCache.txt CMakeFiles cmake_install.cmake

KOKKOS_PATH=${HOME}/Kokkos/kokkos
KOKKOSKERNELS_PATH=${HOME}/Kokkos/kokkos-kernels

KOKKOS_DEVICES="Cuda,OpenMP"
CXX=
KOKKOS_ARCH=
TPLS=
OPTIONS=
CUDA_OPTIONS=

if [[ "${KOKKOS_DEVICES}" == *Cuda* ]]; then
  CXX=${KOKKOS_PATH}/bin/nvcc_wrapper
  KOKKOS_ARCH="BDW,Volta70"
  CUDA_OPTIONS="enable_lambda"
else
  KOKKOS_ARCH="BDW"
  CXX=g++
fi

EXERCISE_DIR=${PWD}
mkdir -p LibInstalls
KOKKOSKERNELS_INSTALL=${EXERCISE_DIR}/LibInstalls/kernels-install
cd LibInstalls

echo ${KOKKOSKERNELS_PATH}/cm_generate_makefile.bash --compiler=${CXX} --with-devices=${KOKKOS_DEVICES} --kokkos-path=${KOKKOS_PATH} --kokkoskernels-path=${KOKKOSKERNELS_PATH} --prefix=${KOKKOSKERNELS_INSTALL} --with-options=${OPTIONS} --with-cuda-options=${CUDA_OPTIONS} --arch=${KOKKOS_ARCH} --with-tpls=${TPLS} --kokkos-make-j=8 --disable-kokkos-tests --disable-tests --disable-examples

${KOKKOSKERNELS_PATH}/cm_generate_makefile.bash --compiler=${CXX} --with-devices=${KOKKOS_DEVICES} --kokkos-path=${KOKKOS_PATH} --kokkoskernels-path=${KOKKOSKERNELS_PATH} --prefix=${KOKKOSKERNELS_INSTALL} --with-options=${OPTIONS} --with-cuda-options=${CUDA_OPTIONS} --arch=${KOKKOS_ARCH} --with-tpls=${TPLS} --kokkos-make-j=8 --disable-kokkos-tests --disable-tests --disable-examples

make install -j8

cd ${EXERCISE_DIR}

echo "KERNELS_INSTALL_PATH = $KOKKOSKERNELS_INSTALL"
if [[ -d "${KOKKOSKERNELS_INSTALL}/lib64" ]]; then
  cmake -DCMAKE_CXX_COMPILER=${CXX} -DKokkosKernels_DIR="${KOKKOSKERNELS_INSTALL}/lib64/cmake/KokkosKernels" .
else
  cmake -DCMAKE_CXX_COMPILER=${CXX} -DKokkosKernels_DIR="${KOKKOSKERNELS_INSTALL}/lib/cmake/KokkosKernels" .
fi
