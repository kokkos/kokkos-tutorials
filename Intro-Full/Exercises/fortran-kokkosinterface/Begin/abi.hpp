#include <stddef.h>
#include <Kokkos_Core.hpp>

extern "C" {

  typedef struct _abi_nd_array_t {
    size_t rank;
    size_t const *dims;
    size_t const *strides;
    void *data;
  } abi_ndarray_t;

}

template <typename DataType>
Kokkos::View<DataType, Kokkos::LayoutStride, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
view_from_ndarray(abi_ndarray_t const &ndarray) {
  size_t dimensions[Kokkos::ARRAY_LAYOUT_MAX_RANK] = {};
  size_t strides[Kokkos::ARRAY_LAYOUT_MAX_RANK] = {};
  using traits = Kokkos::ViewTraits<DataType>;
  using value_type = typename traits::value_type;
  constexpr auto rank = Kokkos::ViewTraits<DataType>::rank;

  if (rank != ndarray.rank) {
    std::cerr << "Requested Kokkos view of rank " << rank << " for ndarray with rank"
    << ndarray.rank << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::copy(ndarray.dims, ndarray.dims + ndarray.rank, dimensions);
  std::copy(ndarray.strides, ndarray.strides + ndarray.rank, strides);

  // clang-format off
  Kokkos::LayoutStride layout{
  dimensions[0], strides[0],
  dimensions[1], strides[1],
  dimensions[2], strides[2],
  dimensions[3], strides[3],
  dimensions[4], strides[4],
  dimensions[5], strides[5],
  dimensions[6], strides[6],
  dimensions[7], strides[7]
  };
  // clang-format on

  return Kokkos::View<DataType, Kokkos::LayoutStride, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
  reinterpret_cast<value_type *>(ndarray.data), layout);
}
