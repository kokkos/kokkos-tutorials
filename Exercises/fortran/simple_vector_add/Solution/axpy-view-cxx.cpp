// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <Kokkos_Core.hpp>
#include <flcl-cxx.hpp>

using view_type = flcl::view_r64_1d_t;

extern "C" {

  void c_axpy_view( view_type **v_y, view_type **v_x, double *alpha ) {
    using flcl::view_from_ndarray;

    view_type y = **v_y;
    view_type x = **v_x;

    double d_alpha = *alpha;
    Kokkos::parallel_for( "axpy", y.extent(0), KOKKOS_LAMBDA( const size_t idx)
    {
      y(idx) += d_alpha * x(idx);
    });

    // make sure data can be reused on host
    Kokkos::fence();

    return;
  }


}
