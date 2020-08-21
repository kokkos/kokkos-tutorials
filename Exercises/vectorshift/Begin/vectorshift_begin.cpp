/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

// EXERCISE Goal: Learn to use PGAS to implement a circular vector shift.

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <assert.h>

#define T int
#define OFFSET 1
#define NUM_SHIFTS 16
#define SIZE 1024

#define swap(a,b,T) T tmp = a; a = b; b=tmp;

//EXERCISE: Change View template parameter to DefaultRemoteMemorySpace in Kokkos::Experimental
using View_t = Kokkos::View<T*>;

//EXERCISE: Change View template parameter to specify a two-dimensional array on the host
using HostView_t = Kokkos::View<T*,Kokkos::HostSpace>;

int main(int argc, char *argv[]) {

  // Excercise: Uncomment networking initialization below
  /*MPI_Init(&argc, &argv);
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
  */

  int k = OFFSET;

  //Excercise: Compute process-local n
  int n = SIZE;


  Kokkos::initialize(argc, argv);
  {
    View_t a("A",n);
    View_t b("B",n);

    //EXERCISE: Add dimension to match remote memory view (1,DIM0,...)
    HostView_t a_h("A_h",n);

    //Initialize one element to non-zero
    Kokkos::deep_copy(a_h,0);

    //EXERCISE: Assign starting shift value to first element of a two-dimensional array
    a_h(0) = 1;

    //Copy to Remote Memory Space
    Kokkos::deep_copy(a, a_h);

    for(int shift = 0; shift < NUM_SHIFTS; ++shift)
    {
      //EXERCISE: Change iteration space to a Range to global array indexing
      Kokkos::parallel_for("Shift",n, KOKKOS_LAMBDA(const int i) {
        int j = i+k; //Shift

        //Excersize: From global array index i, dermine PE and offset within PE
        //Update indexing to two-dimensional indexing
        b(j%n) = a(i);
      });

      //Excercise: Change call to memory space specific fence
      Kokkos::fence();

      swap(a,b,View_t);
    }

    Kokkos::deep_copy(a_h, b);
    //Excersize: Update error check to check if value "1" has been shifter
    //Note: it resides porentially on a different PE
    assert(a_h((NUM_SHIFTS*OFFSET%n)-1)==1);
  }
  Kokkos::finalize();

  // Excersize: Uncomment networking finalization below
  /*
  #ifdef KOKKOS_ENABLE_SHMEMSPACE
  shmem_finalize();
  #endif
  #ifdef KOKKOS_ENABLE_NVSHMEMSPACE
  nvshmem_finalize();
  #endif
  MPI_Finalize();
  */
  return 0;
}
