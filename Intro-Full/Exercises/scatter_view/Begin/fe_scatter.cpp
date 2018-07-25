#include <iostream>

int main() {
  double node_sums[6] = {0, 0, 0, 0, 0, 0};
  int elements_to_nodes[4 * 2] = {0, 1, 4, 3, 1, 2, 5, 4};
  double element_values[2] = {1.0, 2.0};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      int node = elements_to_nodes[i * 4 + j];
      node_sums[node] += element_values[i];
    }
  }
  for (int i = 0; i < 3; ++i) {
    std::cout << "node " << i << " has sum " << node_sums[i] << '\n';
  }
}
