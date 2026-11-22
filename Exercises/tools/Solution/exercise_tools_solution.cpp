// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);

  using h_exec_t = Kokkos::DefaultHostExecutionSpace;
  using exec_t   = Kokkos::DefaultExecutionSpace;

  int N = std::pow(2, 22);  // 4194304

  Kokkos::View<double*> a("A", N);
  Kokkos::View<double*, Kokkos::HostSpace> h_a = Kokkos::create_mirror_view(a);

  Kokkos::Profiling ::pushRegion("Setup");
  Kokkos::parallel_for(
      "Init_A", Kokkos::RangePolicy<h_exec_t>(0, N),
      KOKKOS_LAMBDA(int i) { h_a(i) = i; });

  Kokkos::deep_copy(a, h_a);

  Kokkos::Profiling ::popRegion();
  Kokkos::Profiling ::pushRegion("Iterate");

  for (int r = 0; r < 32; r++) {
    Kokkos::View<double*> tmp("Tmp", N);
    Kokkos::parallel_scan(
        "K_1", Kokkos::RangePolicy<exec_t>(0, N),
        KOKKOS_LAMBDA(int i, double& lsum, bool f) {
          if (f) tmp(i) = lsum;
          lsum += a(i);
        });
    double sum;
    Kokkos::parallel_reduce(
        "K_2", N, KOKKOS_LAMBDA(int i, double& lsum) { lsum += tmp(i); }, sum);
  }
  Kokkos::Profiling ::popRegion();
}
