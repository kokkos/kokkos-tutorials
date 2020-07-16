#include <mpi.h>
#include <cassert>
#include <sstream>
#include <vector>
#include <cstdio>
#include <Kokkos_Core.hpp>

void extract_and_sort_ranks(
    Kokkos::View<int*> destination_ranks,
    Kokkos::View<int*> permutation,
    std::vector<int>& unique_ranks,
    std::vector<int>& offsets,
    std::vector<int>& counts) {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  auto n = destination_ranks.extent(0);
  using ST = decltype(n);
  Kokkos::View<int*> tmp_ranks("tmp ranks", destination_ranks.extent(0));
  Kokkos::deep_copy(tmp_ranks, destination_ranks);
  int offset = 0;
  // this implements a "sort" which is O(N * R) where (R) is
  // the total number of unique destination ranks.
  // it performs better than other algorithms in
  // the case when (R) is small, but results may vary
  while (true) {
    int next_biggest_rank;
    Kokkos::parallel_reduce("find next biggest rank", n, KOKKOS_LAMBDA(ST i, int& local_max) {
      auto r = tmp_ranks(i);
      local_max = (r > local_max) ? r : local_max; 
    }, Kokkos::Max<int>(next_biggest_rank));
    if (next_biggest_rank == -1) break;
    unique_ranks.push_back(next_biggest_rank);
    offsets.push_back(offset);
    Kokkos::View<int> total("total");
    Kokkos::parallel_scan("process biggest rank items", n,
    KOKKOS_LAMBDA(ST i, int& index, const bool last_pass) {
      if (last_pass && (tmp_ranks(i) == next_biggest_rank)) {
        permutation(i) = index + offset;
      }
      if (tmp_ranks(i) == next_biggest_rank) ++index;
      if (last_pass) {
        if (i + 1 == tmp_ranks.extent(0)) {
          total() = index;
        }
        if (tmp_ranks(i) == next_biggest_rank) {
          tmp_ranks(i) = -1;
        }
      }
    });
    auto host_total = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), total);
    auto count = host_total();
    counts.push_back(count);
    offset += count;
  }
}

void main2() {
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  // this input describes a quadrilateral strip mesh where
  // every MPI rank owns one quadrilateral, and more importantly
  // it owns the two nodes on the "left" of the quadrilateral (nodes 0 and 3).
  // the special case of the "rightmost" quadrilateral owns
  // all of its nodes.
  std::size_t n = 4;
  Kokkos::View<int*> destination_ranks("destination ranks", n);
  Kokkos::View<int*> destination_indices("destination indices", n);
  Kokkos::View<double*> values("values", n);
  Kokkos::deep_copy(values, 1.0);
  std::vector<int> host_ranks, host_indices;
  if (mpi_rank + 1 == mpi_size) {
    host_ranks = {mpi_rank, mpi_rank, mpi_rank, mpi_rank};
    host_indices = {0, 1, 2, 3};
  } else {
    host_ranks = {mpi_rank, mpi_rank + 1, mpi_rank + 1, mpi_rank};
    host_indices = {0, 0, 3, 3};
  }
  Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
  host_ranks2(host_ranks.data(), n);
  Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
  host_indices2(host_indices.data(), n);
  Kokkos::deep_copy(destination_ranks, host_ranks2);
  Kokkos::deep_copy(destination_indices, host_indices2);
  // Step 1: sort by rank, extract auxiliary info
  Kokkos::View<int*> permutation("permutation", n);
  std::vector<int> destinations;
  std::vector<int> dest_offsets;
  std::vector<int> dest_counts;
  extract_and_sort_ranks(destination_ranks, permutation, destinations, dest_offsets, dest_counts);
  // Step 2: use graph communicators to determine incoming messages
  auto num_destinations = int(destinations.size());
  int mpi_graph_num_sources = 1;
  int mpi_graph_sources[1] = {mpi_rank};
  int mpi_graph_degrees[1] = {num_destinations};
  int* mpi_graph_destinations = destinations.data();
  int mpi_graph_reorder = 0;
  MPI_Comm graph_comm;
  MPI_Dist_graph_create(MPI_COMM_WORLD, mpi_graph_num_sources, mpi_graph_sources,
      mpi_graph_degrees, mpi_graph_destinations, MPI_UNWEIGHTED, MPI_INFO_NULL,
      mpi_graph_reorder, &graph_comm);
  int num_sources;
  int is_weighted;
  MPI_Dist_graph_neighbors_count(graph_comm, &num_sources, &num_destinations, &is_weighted);
  auto sources = std::vector<int>(std::size_t(num_sources));
  MPI_Dist_graph_neighbors(graph_comm, num_sources, sources.data(), MPI_UNWEIGHTED,
      num_destinations, destinations.data(), MPI_UNWEIGHTED);
  auto src_counts = std::vector<int>(std::size_t(num_sources));
  MPI_Neighbor_alltoall(dest_counts.data(), 1, MPI_INT, src_counts.data(), 1, MPI_INT,
      graph_comm);
  // Step 3: set up metadata and buffer for incoming
  auto src_offsets = std::vector<int>(std::size_t(num_sources + 1));
  src_offsets[0] = 0;
  for (std::size_t i = 0; i < std::size_t(num_sources); ++i) {
    src_offsets[i + 1] = src_offsets[i] + src_counts[i];
  }
  int nrecvd = src_offsets[std::size_t(num_sources)];
  // Step 4: permute message data
  Kokkos::View<int*> msg_indices_by_dest("indices by dest", n);
  Kokkos::View<double*> msg_values_by_dest("values by dest", n);
  Kokkos::parallel_for("permute", n, KOKKOS_LAMBDA(int i) {
    msg_indices_by_dest[permutation[i]] = destination_indices[i];
    msg_values_by_dest[permutation[i]] = values[i];
  });
  // Step 5: use MPI_Neighbor_alltoallv to exchange the data
  Kokkos::View<int*> msg_indices_by_src("indices by src", nrecvd);
  Kokkos::View<double*> msg_values_by_src("values by src", nrecvd);
  MPI_Neighbor_alltoallv(msg_values_by_dest.data(), dest_counts.data(), dest_offsets.data(),
      MPI_DOUBLE, msg_values_by_src.data(), src_counts.data(), src_offsets.data(), MPI_DOUBLE,
      graph_comm);
  MPI_Neighbor_alltoallv(msg_indices_by_dest.data(), dest_counts.data(), dest_offsets.data(),
      MPI_INT, msg_indices_by_src.data(), src_counts.data(), src_offsets.data(), MPI_INT,
      graph_comm);
  MPI_Comm_free(&graph_comm);
  // Step 6: not part of communication, but take action based on the received "messages"
  Kokkos::deep_copy(values, 0.0);
  Kokkos::parallel_for("add contributions", nrecvd, KOKKOS_LAMBDA(int i) {
      // note: consider using a ScatterView for this algorithm
      Kokkos::atomic_add(&values[msg_indices_by_src[i]], msg_values_by_src[i]);
  });
  // Step 7: copy values to the host and print so the user knows their algorithm worked okay
  auto host_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  std::stringstream ss;
  ss << "rank " << mpi_rank << " values: {";
  for (int i = 0; i < n; ++i) {
    ss << host_values[i];
    if (i + 1 < n) ss << ", ";
  }
  ss << "}";
  auto s = ss.str();
  std::printf("%s\n", s.c_str());
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  main2();
  Kokkos::finalize();
  MPI_Finalize();
}
