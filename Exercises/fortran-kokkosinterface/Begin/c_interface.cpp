#include "abi.hpp"

extern "C" {

  void c_kokkos_initialize() { Kokkos::initialize(); }

  void c_kokkos_finalize( void ) { Kokkos::finalize(); }

  void c_axpy_kokkos( double &alpha, abi_ndarray_t &nd_array_x, abi_ndarray_t &nd_array_y ) {

    // TODO: use view_from_ndarray to instantiate Views array_x and array_y
    // from nd_array_x and nd_array_y that were passed over the ABI
    // TIP: use auto as the type for array_x and array_y

    // TODO: uncomment the parallel_for once array_x and array_y
    // are instantiated
    // Kokkos::parallel_for(nd_array_x.dims[0],
    //     KOKKOS_LAMBDA (const size_t ii) { array_y(ii) += alpha * array_x(ii); }
    // );
  }

}
