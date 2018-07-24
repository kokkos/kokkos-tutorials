/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

// Exercise
// 1 insert ids into global to local map and check for failure
// 2 iterate over the map

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
      // TODO insert ids into g2l
    }, errors);

    if (errors > 0 ) {
      printf("Exceeded UnorderedMap capacity\n");
    }

    // Iterate over g2l map

    errors = 0;

    Kokkos::parallel_reduce(g2l.capacity(), KOKKOS_LAMBDA( int x, int & local_errors ) {
      // TODO check if valid at first
        const auto gid = g2l.key_at(x);
        const auto lid = g2l.value_at(x);

        if (l2g(lid) != gid) {
          local_errors += 1;
        }
    }, errors);


    if (errors > 0 ) {
      printf("UnorderedMap has incorrect values\n");
    }

  }
  Kokkos::finalize();

  return 0;
}

