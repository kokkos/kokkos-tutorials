#include <Kokkos_ScatterView.hpp>

KOKKOS_INLINE_FUNCTION int foo(int i) { return i; }
KOKKOS_INLINE_FUNCTION double bar(int) { return 55.0; }

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  Kokkos::View<double*> results("results", 1);
  Kokkos::Experimental::ScatterView<double*> scatter(results);
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(int input_i) {
    auto access = scatter.access();
    auto result_i = foo(input_i);
    auto contribution = bar(input_i);
    access(result_i) += contribution;
  });
  Kokkos::Experimental::contribute(results, scatter);
}
