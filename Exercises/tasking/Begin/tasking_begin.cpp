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

// EXERCISE 8 Goal:
//   Convert the serial Fibonacci code into a task that creates new tasks
//   for each recursive subproblem.
//     1. Implement the FibonacciTask task functor
//     2. Spawn an instance of that task from the host in `main()`
//     3. Wait for the scheduler to make that task's result ready (in `main()`)
//     4. Try adding priorities to different spawns and respawns to see how
//        this affects execution time.
//     5. Try out different schedulers and see how this affects execution time.

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>

#include <Kokkos_Core.hpp>

//==============================================================================

// Use this to estimate the size of the memory pool for a fibonacci problem with
// argument `n`
size_t
estimate_required_memory(int n)
{
  assert(n >= 0);
  auto nl = static_cast<size_t>(n);
  return (nl + 1) * (nl + 1) * 2000;
}

//==============================================================================

// We want to implement a task-parallel version of this serial algorithm:
int fib_serial(int n)
{
  if(n < 2) return n;
  else {
    return fib_serial(n-1) + fib_serial(n-2);
  }
}

//==============================================================================

struct FibonacciTask
{
  // EXERCISE:  Write a Kokkos task functor to recursively compute the `n`th
  //            Fibonacci number, keeping in mind that the computation of the
  //            `n-1`th Fibonacci number and the `n-2`th Fibonacci number can
  //            be computed simultaneously.
  // ...
};

//==============================================================================

int main(int argc, char* argv[])
{
  int n = 10; // Fib number to compute

  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    if (std::string(argv[i]) == "-N") {
      n = std::atoi(argv[++i]);
      printf("  User N is %d\n", n);
    }
    else if(std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help") {
      printf("  Fibonacci Options:\n");
      printf("  -N <int>:      Which Fibonacci number to compute (default: 10)\n");
      printf("  --help (-h):   print this message\n\n");
      std::exit(1);
    }
  }

  // Check argument.
  if(n < 0) {
    printf("Argument to Fibonacci must be non-negative.\n");
    std::exit(1);
  }

  Kokkos::initialize(argc, argv);
  {
    using execution_space = Kokkos::DefaultExecutionSpace;
    using scheduler_type = Kokkos::TaskScheduler<execution_space>;
    using memory_space = typename scheduler_type::memory_space;
    using memory_pool = typename scheduler_type::memory_pool;

    auto mpool = memory_pool(memory_space{}, estimate_required_memory(n));
    auto scheduler = scheduler_type(mpool);

    Kokkos::BasicFuture<int, scheduler_type> result;

    Kokkos::Timer timer;
    {

      // EXERCISE: Spawn a task from the host using the FibonacciTask functor to
      //           recursively compute the `n`th Fibonacci number and assign the
      //           resulting future to `result`
      // ...


      // EXERCISE: Wait on the scheduler to ensure `result` is ready
      // ...
      
    }
    auto time = timer.seconds();

    // Output results
    if(!result.is_null() && result.is_ready()) {
      auto result_serial = fib_serial(n);
      if(result.get() == result_serial) {
        printf("  Success! Fibonacci(%d) = %d\n", n, result.get());
      }
      else {
        printf(
          "  Error! Task result of Fibonacci(%d) was %d, but serial result was %d\n",
          n, result.get(), result_serial
        );
      }
    }
    else {
      printf("  Error! Result of Fibonacci(%d) is not ready\n", n);
    }

    // Print the timing results
    printf("  Task computation executed in %g s\n", time);
  }
  Kokkos::finalize();

  return 0;
}
