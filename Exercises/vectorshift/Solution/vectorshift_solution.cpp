// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// EXERCISE Goal: Learn to use PGAS to implement a circular vector shift.

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <assert.h>

#define T int
#define OFFSET 1
#define NUM_SHIFTS 16
#define SIZE 1024

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteView_t = Kokkos::View<T**,RemoteSpace_t>;
using HostView_t = Kokkos::View<T **,Kokkos::HostSpace>;

#define swap(a,b,T) T tmp = a; a = b; b=tmp;

int main(int argc, char *argv[]) {
  
  // Init
  MPI_Init(&argc, &argv);

#ifdef KOKKOS_ENABLE_SHMEMSPACE
  shmem_init();
#endif
#ifdef KOKKOS_ENABLE_NVSHMEMSPACE
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  mpi_comm = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif
  
  int myPE, numPEs;
  MPI_Comm_rank(MPI_COMM_WORLD, &myPE);
  MPI_Comm_size(MPI_COMM_WORLD, &numPEs);
  
  int k = OFFSET;
  int n = SIZE;

  //Excercise: Compute process-local n
  int myN = n / numPEs;

  k = (k>myN)?myN:k;

  Kokkos::initialize(argc, argv);
  {
    RemoteView_t a("A",numPEs,myN);
    RemoteView_t b("B",numPEs,myN);

    //Adding dimension to match remote memory view (1,DIM0,...)
    HostView_t a_h("A_h",1,myN);

    //Initialize to Zero
    Kokkos::deep_copy(a_h,0);

    //Initialize one element to non-zero
    a_h(0,0) = 1; 

    //Copy to Remote Memory Space
    Kokkos::Experimental::deep_copy(a, a_h);

    for(int shift = 0; shift < NUM_SHIFTS; ++shift)
    {   
      //Iteration space over global array
      Kokkos::parallel_for("Shift",Kokkos::RangePolicy<>(myPE*myN,(myPE+1)*myN), 
      KOKKOS_LAMBDA(const int i) { 
        int j = i+k; //Shift

        //From global array index i, dermining PE and offset within PE
        //using two-dimensional indexing
        b((j / myN) % numPEs, j%myN) = (T) a(myPE, i);
      });

      RemoteSpace_t().fence();

      swap(a,b,RemoteView_t);
    }
    //Copy back to Host memory space  
    Kokkos::Experimental::deep_copy(a_h, b);

    // Correctness check on corresponding PE
    if (myPE == NUM_SHIFTS*OFFSET / myN ){
      assert(a_h(0,(NUM_SHIFTS*OFFSET % myN)-1)==1);
    }
  }

  Kokkos::finalize();
  #ifdef KOKKOS_ENABLE_SHMEMSPACE
  shmem_finalize();
  #endif
  #ifdef KOKKOS_ENABLE_NVSHMEMSPACE
  nvshmem_finalize();
  #endif
  MPI_Finalize();
  return 0;
}
