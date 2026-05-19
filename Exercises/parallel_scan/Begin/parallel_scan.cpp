// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project


#include <Kokkos_Core.hpp>
#include <iostream>

template <typename ValueType> struct Factorial {
  /* EXERCISE */
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
