//@HEADER
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
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <iostream>

template <typename ValueType>
struct GeometricMean {
  using value_type = ValueType;

  GeometricMean(Kokkos::View<value_type *> view)
      : m_view(view), n(view.size()) {}

  KOKKOS_FUNCTION void init(value_type &value) const { value = 1.; }
  KOKKOS_FUNCTION void join(value_type &lhs, const value_type &rhs) const {
    lhs *= rhs;
  }
  KOKKOS_FUNCTION void final(value_type &value) const {
    value = Kokkos::pow(value, 1. / n);
  }

  KOKKOS_FUNCTION void operator()(int i, value_type &update) const {
    update *= m_view(i);
  }

private:
  Kokkos::View<value_type *> m_view;
  int n;
};

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int n = 10;
    Kokkos::View<double *> view("view", n);
    Kokkos::parallel_for(
        n, KOKKOS_LAMBDA(int i) { view(i) = 1 + i / 10.; });

    double result;
    Kokkos::parallel_reduce(n, GeometricMean{view}, result);

    auto host_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
    double reference = 1.;
    for (int i = 0; i < n; ++i) {
      reference *= host_view(i);
    }

    std::cout << "geometric mean: " << result << ' '
              << "reference: " << std::pow(reference, 1. / n) << '\n';
  }
  Kokkos::finalize();
}
