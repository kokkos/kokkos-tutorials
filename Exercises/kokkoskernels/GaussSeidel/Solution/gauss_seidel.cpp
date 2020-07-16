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
// Questions Contact Brian Kelley (bmkelle@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

// EXERCISE
//   - Goal: Run parallel Gauss-Seidel as an iterative solver on a diagonally dominant system.

#include "Kokkos_Core.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosKernels_Handle.hpp"
#include "KokkosKernels_IOUtils.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_gauss_seidel.hpp"
#include "KokkosBlas1_nrm2.hpp"

using Scalar  = default_scalar;
using Mag     = Kokkos::ArithTraits<Scalar>::mag_type;
using Ordinal = default_lno_t;
using Offset  = default_size_type;
using Layout  = default_layout;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = typename ExecSpace::memory_space;
using Device  = Kokkos::Device<ExecSpace, MemSpace>;
using Handle  = KokkosKernels::Experimental::
  KokkosKernelsHandle<Offset, Ordinal, default_scalar, ExecSpace, MemSpace, MemSpace>;
using Matrix  = KokkosSparse::CrsMatrix<Scalar, Ordinal, Device, void, Offset>;
using Vector  = typename Matrix::values_type;

constexpr Ordinal numRows = 10000;
const Scalar one = Kokkos::ArithTraits<Scalar>::one();
const Mag magOne = Kokkos::ArithTraits<Mag>::one();
//Solve tolerance
const Mag tolerance = 1e-6 * magOne;

//Helper to print out colors in the shape of the grid
int main(int argc, char* argv[])
{
  Kokkos::initialize();
  {
    //Generate a square, strictly diagonally dominant, but nonsymmetric matrix on which Gauss-Seidel should converge.
    //Get approx. 20 entries per row
    //Diagonals are 2x the absolute sum of all other entries.
    Offset nnz = numRows * 20;
    Matrix A = KokkosKernels::Impl::kk_generate_diagonally_dominant_sparse_matrix<Matrix>(numRows, numRows, nnz, 2, 100, 1.05 * one);
    std::cout << "Generated a matrix with " << numRows << " rows/cols, and " << nnz << " entries.\n";
    //Create a kernel handle, then a Gauss-Seidel handle with the default algorithm
    Handle handle;
//- EXERCISE: Create a GS subhandle with the default algorithm (hint: create_gs_handle)
    handle.create_gs_handle(KokkosSparse::GS_DEFAULT);
//- EXERCISE: Do symbolic setup (hint: use numRows, A.graph.row_map and A.graph.entries. The matrix is square and nonsymmetric)
    KokkosSparse::Experimental::gauss_seidel_symbolic(&handle, numRows, numRows, A.graph.row_map, A.graph.entries, false);
    //Set up Gauss-Seidel for the matrix values (numeric)
    //Another matrix with the same sparsity pattern could re-use the handle and symbolic phase, and only call numeric.
//- EXERCISE: Do numeric setup (hint: same interface as symbolic, except it also uses A.values)
    KokkosSparse::Experimental::gauss_seidel_numeric(&handle, numRows, numRows, A.graph.row_map, A.graph.entries, A.values, false);
    //Now, preconditioner is ready to use. Set up an unknown vector (uninitialized) and randomized right-hand-side vector.
    Vector x(Kokkos::ViewAllocateWithoutInitializing("x"), numRows);
    Vector b(Kokkos::ViewAllocateWithoutInitializing("b"), numRows);
    Vector res(Kokkos::ViewAllocateWithoutInitializing("res"), numRows);
    auto bHost = Kokkos::create_mirror_view(b);
    for(Ordinal i = 0; i < numRows; i++)
      bHost(i) = 3 * ((one * rand()) / RAND_MAX);
    Kokkos::deep_copy(b, bHost);
    //Measure initial residual norm ||Ax - b||, where x is 0
    Mag initialRes = KokkosBlas::nrm2(b);
    Mag scaledResNorm = magOne;
    bool firstIter = true;
    //Iterate until reaching the tolerance
    int numIters = 0;
    while(scaledResNorm > tolerance)
    {
//- EXERCISE: Apply Gauss-Seidel to the system Ax = b
//    * Hint: firstIter will be true for the first sweep, and false for all others. Use this fact to:
//      - zero out x on the first sweep only, since it was not initialized above.
//      - set update_y_vector true on the first sweep only, since GS has not seen that right-hand side before.
//        After that, b does not change so it should be false.
//    * Perform one forward sweep.
//    * Use damping factor (omega) of 1.0.
      KokkosSparse::Experimental::forward_sweep_gauss_seidel_apply(
          &handle, numRows, numRows,
          A.graph.row_map, A.graph.entries, A.values,
          x, b, firstIter, firstIter, one, 1);
      firstIter = false;
      //Now, compute the new residual norm using SPMV
      Kokkos::deep_copy(res, b);
      //Compute res := Ax - res (since res is now equal to b, this is Ax - b)
      KokkosSparse::spmv("N", one, A, x, -one, res);
      //Recompute the scaled norm
      scaledResNorm = KokkosBlas::nrm2(res) / initialRes;
      numIters++;
      std::cout << "Iteration " << numIters << " scaled residual norm: " << scaledResNorm << '\n';
    }
    std::cout << "SUCCESS: converged in " << numIters << " iterations.\n";
  }
  Kokkos::finalize();
  return 0;
}

