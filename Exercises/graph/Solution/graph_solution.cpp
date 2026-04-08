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

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <Kokkos_Core.hpp>

#include <Kokkos_Random.hpp>
#include <Kokkos_Graph.hpp>

template<class T>
constexpr bool is_view_v = false;

template<class T, class ... Args>
constexpr bool is_view_v<Kokkos::View<T, Args...>> = true;

template<class T>
concept view = is_view_v<T>;

using policy_t = Kokkos::RangePolicy<>;

template<view D, view P>
void init(D data, P pack_ids) {
  Kokkos::parallel_for("Init Data", policy_t(0, data.extent(0)),
    KOKKOS_LAMBDA(int i) { data(i) = i; });
  Kokkos::Random_XorShift64_Pool<> rand_pool64(5374857);
  Kokkos::fill_random(pack_ids, rand_pool64, data.extent(0));
}

// CUDA does not support auto return type from functions
// which create host device lambdas
template<view D, view P, view B>
struct pack_functor {
  D data;
  P pack_ids;
  B buffer;
  KOKKOS_FUNCTION void operator() (int i) const {
    buffer(i) = data(pack_ids(i));
  }
};

template<class GraphNode, view D, view P, view B>
auto pack(GraphNode node, D data, P pack_ids, B buffer) {
  return node.then_parallel_for("Pack One", policy_t(0, pack_ids.extent(0)),
    pack_functor{data, pack_ids, buffer});
}

template<view Dest, view Src>
struct copy_functor {
  Dest d;
  Src s;
  KOKKOS_FUNCTION void operator() (int i) const {
    d(i) = s(i);
  }
};

template<class GraphNode, view R, view S>
auto transfer(GraphNode node, R recv, S send) {
  auto temp_node = node.then_parallel_for("DeepCopy", policy_t(0, recv.extent(0)),
    copy_functor{recv, send});
  return temp_node.then_host("HostThing", [=]() { printf("HostTransfer %p %p\n",recv.data(), send.data()); });
}

template<class GraphNode, view D, view B>
auto unpack(GraphNode node, D data, B buffer) {
  return node.then_parallel_for("DeepCopy", policy_t(0, buffer.extent(0)),
    copy_functor{data, buffer});
}

void mpi_style_iteration(int num_elements, int num_mpi_neighs, int num_sendrecv, int num_repeat) {
  Kokkos::View<double*> data("Data", num_elements + num_sendrecv);
  Kokkos::View<double**, Kokkos::LayoutRight> send_buffer("SendBuf", num_mpi_neighs, num_sendrecv);
  Kokkos::View<double**, Kokkos::LayoutRight> recv_buffer("RecvBuf", num_mpi_neighs, num_sendrecv);
  Kokkos::View<double**, Kokkos::LayoutRight> pack_ids("PackIDS", num_mpi_neighs, num_sendrecv);
  init(data, pack_ids);

  Kokkos::Timer timer;
  Kokkos::Experimental::Graph graph;

  for(int neigh = 0; neigh < num_mpi_neighs; neigh++) {
    auto my_pack_ids = Kokkos::subview(pack_ids, neigh, Kokkos::ALL());
    auto send_buf = Kokkos::subview(send_buffer, neigh, Kokkos::ALL());
    auto recv_buf = Kokkos::subview(recv_buffer, neigh, Kokkos::ALL());
    auto node1 = pack(graph.root_node(), data, my_pack_ids, send_buf);
    auto node2 = transfer(node1, recv_buf, send_buf);
    auto node3 = unpack(node2, Kokkos::subview(data, Kokkos::pair{num_elements, (int)data.extent(0)}), recv_buf);
  }
  graph.instantiate();
  Kokkos::fence();
  printf("Graph Create Done\n");

  double time_create = timer.seconds();
  timer.reset();
  for(int r=0; r < num_repeat; r++) {
    graph.submit();
    Kokkos::fence();
  }
  double time = timer.seconds();
  printf("Graph Runtime: %lf %lf\n",time*1000, time_create*1000);
}


int main( int argc, char* argv[] )
{
  int64_t N = 20000;  // number of elements
  int neighs = 6;       // number of neighbors
  int num_send = 5000; // number of elements to send/recv
  int nrepeat = 10;     // number of repeats of the test

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( strcmp( argv[ i ], "-N" ) == 0 ) {
      N = atoi( argv[ ++i ] );
      printf( "  User N is %lld\n", N );
    }
    else if ( strcmp( argv[ i ], "-neighs" ) == 0 ) {
      neighs = atoi( argv[ ++i ] );
    }
    else if ( strcmp( argv[ i ], "-nsend" ) == 0 ) {
      num_send = atoi( argv[ ++i ] );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  -N <int>:       number of elements (default: 20000)\n" );
      printf( "  -neighs <int>:  number of neighbors (default: 6)\n" );
      printf( "  -nsend <int>:   number of send/recv elements (default: 5000)\n" );
      printf( "  -nrepeat <int>: number of repetitions (default: 10)\n" );
      printf( "  -help (-h):     print this message\n\n" );
      exit( 1 );
    }
  }


  Kokkos::initialize( argc, argv );
  {
    printf("Execute with %lld %i %i %i\n",N, neighs, num_send, nrepeat);
    mpi_style_iteration(N, neighs, num_send, nrepeat);
  }
  Kokkos::finalize();

  return 0;
}

