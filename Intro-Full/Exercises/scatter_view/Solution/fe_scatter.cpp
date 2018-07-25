#include <iostream>
#include <Kokkos_ScatterView.hpp>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);

  double node_sums_c[6] = {0, 0, 0, 0, 0, 0};
  Kokkos::View<double*, Kokkos::HostSpace> node_sums_host(node_sums_c, 6);
  Kokkos::View<double*> node_sums("node sums", 6);
  Kokkos::deep_copy(node_sums, node_sums_host);

  int elements_to_nodes_c[4 * 2] = {0, 1, 4, 3, 1, 2, 5, 4};
  Kokkos::View<int*[4], Kokkos::HostSpace> elements_to_nodes_host(
      elements_to_nodes_c, 2);
  Kokkos::View<int*[4]> elements_to_nodes(
      "elements to nodes", 2);
  Kokkos::deep_copy(elements_to_nodes, elements_to_nodes_host);

  double element_values_c[2] = {1.0, 2.0};
  Kokkos::View<double*, Kokkos::HostSpace> element_values_host(
      element_values_c, 2);
  Kokkos::View<double*> element_values("element values", 2);
  Kokkos::deep_copy(element_values, element_values_host);

  Kokkos::Experimental::ScatterView<double*> scatter(node_sums);

  Kokkos::parallel_for("scatter loop", 2, KOKKOS_LAMBDA(int i) {
    auto node_sums_access = scatter.access();
    for (int j = 0; j < 4; ++j) {
      int node = elements_to_nodes(i, j);
      node_sums_access(node) += element_values(i);
    }
  });
  Kokkos::Experimental::contribute(node_sums, scatter);
  Kokkos::deep_copy(node_sums_host, node_sums);
  for (int i = 0; i < 6; ++i) {
    std::cout << "node " << i << " has sum " << node_sums_host[i] << '\n';
  }
}
