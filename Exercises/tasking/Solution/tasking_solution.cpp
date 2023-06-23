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

template <class Scheduler>
struct FibonacciTask
{
  using value_type = int;
  using future_type = Kokkos::BasicFuture<int, Scheduler>;

  int n;
  future_type fn_1;
  future_type fn_2;

  KOKKOS_INLINE_FUNCTION
  explicit
  FibonacciTask(int num) noexcept
    : n(num)
  { }

  template <class TeamMember>
  KOKKOS_INLINE_FUNCTION
  void operator()(TeamMember& member, int& result) {
    auto& scheduler = member.scheduler();
    if(n < 2) {
      // this is the recursive base case
      result = n;
    }
    else if(!fn_1.is_null() && !fn_2.is_null()) {
      // We only get here after respawn, so just set the result
      result = fn_1.get() + fn_2.get();
    }
    else {
      // Spawn child tasks for the subproblems
      fn_1 = Kokkos::task_spawn(
        Kokkos::TaskSingle(scheduler),
        FibonacciTask{n-1}
      );
      fn_2 = Kokkos::task_spawn(
        Kokkos::TaskSingle(scheduler),
        FibonacciTask{n-2}
      );
      
      // Create an aggregate predecessor for our respawn
      Kokkos::BasicFuture<void, Scheduler> fib_array[] = { fn_1, fn_2 };
      auto f_all = scheduler.when_all(fib_array, 2);

      // Respawn this task with `f_all` as a predecessor
      Kokkos::respawn(this, f_all);
    }
  }

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
    using scheduler_type = Kokkos::TaskScheduler<Kokkos::DefaultExecutionSpace>;
    using memory_space = typename scheduler_type::memory_space;
    using memory_pool = typename scheduler_type::memory_pool;

    auto mpool = memory_pool(memory_space{}, estimate_required_memory(n));
    auto scheduler = scheduler_type(mpool);

    Kokkos::BasicFuture<int, scheduler_type> result;

    Kokkos::Timer timer;
    {
      // launch the root task from the host
      result =
        Kokkos::host_spawn(
          Kokkos::TaskSingle(scheduler),
          FibonacciTask<scheduler_type>{n}
        );

      // wait on all tasks submitted to the scheduler to be done
      Kokkos::wait(scheduler);
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
