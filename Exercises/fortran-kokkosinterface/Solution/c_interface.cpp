#include "abi.hpp"

extern "C" {

  void c_kokkos_initialize() { Kokkos::initialize(); }

  void c_kokkos_finalize( void ) { Kokkos::finalize(); }

  void c_axpy_kokkos( double &alpha, abi_ndarray_t &nd_array_x, abi_ndarray_t &nd_array_y ) {

    auto array_x = view_from_ndarray<double*>(nd_array_x);
    auto array_y = view_from_ndarray<double*>(nd_array_y);

    Kokkos::parallel_for(nd_array_x.dims[0],
        KOKKOS_LAMBDA (const size_t ii) { array_y(ii) += alpha * array_x(ii); }
    );
  }

}
