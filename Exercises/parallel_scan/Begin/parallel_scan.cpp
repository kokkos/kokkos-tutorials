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

template <typename ValueType> struct Factorial {
  using value_type = ValueType;
  
  Factorial(Kokkos::View<value_type *> view) : m_view(view) {}
  
  // EXERCISE: Implement the init method
  // void init(...) {...}
  
  // EXERCISE: Implement the join method
  // void join(...) {...}

  // EXERCISE: Implement the operator() method
  // void operator()(...) const {...}

private:
  Kokkos::View<value_type *> m_view;
};

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int n = 10;
    Kokkos::View<double *> view("view", n);

    /* EXERCISE */
    // Kokkos::parallel_scan(n, Factorial(view));

    auto host_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
    Kokkos::View<double *, Kokkos::HostSpace> reference_view("reference", n);
    double reference = 1.;
    for (int i = 0; i < n; ++i) {
      reference_view(i) = reference;
      reference *= i + 1;
    }

    std::cout << "Factorial: \n";
    for (int i = 0; i < n; ++i)
      std::cout << i << ": " << host_view(i) << " should be "
                << reference_view(i) << '\n';
  }

  Kokkos::finalize();
}
