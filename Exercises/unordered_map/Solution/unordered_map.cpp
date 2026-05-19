// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>

int main(int argc, char *argv[])
{
  Kokkos::initialize( argc, argv );
  {
    int num_ids = 100000;
    int map_capacity = num_ids;

    if (argc > 1) {
      num_ids = atoi(argv[1]);
    }
    if (argc > 2 ) {
      map_capacity = atoi(argv[2]);
    }

    Kokkos::View<uint64_t*> l2g("local_2_global", num_ids);

    // Generate fake global ids from the local id
    Kokkos::parallel_for( num_ids, KOKKOS_LAMBDA( int i ) {

      constexpr uint64_t byte = 0xFFull;
      uint64_t tmp = static_cast<uint64_t>(~i);

      l2g(i) = (tmp & (byte <<  0)) << 56 // 56 byte[7]
             | (tmp & (byte <<  8)) << 40 // 48 byte[6]
             | (tmp & (byte << 16)) << 24 // 40 byte[5]
             | (tmp & (byte << 24)) <<  8 // 32 byte[4]
             | (tmp & (byte << 32)) >>  8 // 24 byte[3]
             | (tmp & (byte << 40)) >> 24 // 16 byte[2]
             | (tmp & (byte << 48)) >> 40 //  8 byte[1]
             | (tmp & (byte << 56)) >> 56 //  0 byte[0]
             ;
    });

    Kokkos::UnorderedMap<uint64_t,int32_t> g2l(map_capacity);


    int errors = 0;

    // Fill global to local map
    Kokkos::parallel_reduce(num_ids, KOKKOS_LAMBDA( int i, int & local_errors ) {
      auto result = g2l.insert( l2g[i], i );
      if (result.failed()) {
        local_errors += 1;
      }
    }, errors);

    if (errors > 0 ) {
      printf("Exceeded UnorderedMap capacity\n");
    }

    // Iterate over g2l map

    errors = 0;

    Kokkos::parallel_reduce(g2l.capacity(), KOKKOS_LAMBDA( int x, int & local_errors ) {

      if( g2l.valid_at(x) ) {
        const auto gid = g2l.key_at(x);
        const auto lid = g2l.value_at(x);

        if (l2g(lid) != gid) {
          local_errors += 1;
        }
      }
    }, errors);


    if (errors > 0 ) {
      printf("UnorderedMap has incorrect values\n");
    }

  }
  Kokkos::finalize();

  return 0;
}

