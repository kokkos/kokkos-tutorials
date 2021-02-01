#include<Kokkos_Core.hpp>
#include<Kokkos_ScatterView.hpp>

// Scatter Add algorithm using atomics
double scatter_view_loop(Kokkos::View<int**> v, 
		 Kokkos::View<int*> r) {
  Kokkos::Experimental::ScatterView<int*> results(r);
  Kokkos::Timer timer;

  results.reset();
  // Run Atomic Loop not r is already using atomics by default
  Kokkos::parallel_for("Atomic Loop", v.extent(0), 
    KOKKOS_LAMBDA(const int i) {
    auto access = results.access();
    for(int j=0; j<v.extent(1); j++)
      access(v(i,j))+=1;
  });
  Kokkos::Experimental::contribute(r,results);
  // Wait for Kernel to finish before timing
  Kokkos::fence();
  double time = timer.seconds();
  return time;
}

// Scatter Add algorithm using atomics
double atomic_loop(Kokkos::View<int**> v, 
		 Kokkos::View<int*,Kokkos::MemoryTraits<Kokkos::Atomic>> r) {
  Kokkos::Timer timer;
  // Run Atomic Loop not r is already using atomics by default
  Kokkos::parallel_for("Atomic Loop", v.extent(0), 
    KOKKOS_LAMBDA(const int i) {
    for(int j=0; j<v.extent(1); j++)
      r(v(i,j))++;
  });
  // Wait for Kernel to finish before timing
  Kokkos::fence();
  return timer.seconds();
}

#if defined(KOKKOS_ENABLE_OPENMP)
// Scatter Add algorithm using data replication
double openmp_loop(Kokkos::View<int**,Kokkos::HostSpace> v, 
		 Kokkos::View<int*,Kokkos::HostSpace> r) {
  // Not timing creation of duplicated arrays, assume you can reuse
  Kokkos::View<int**,Kokkos::HostSpace> results("Rdup",Kokkos::OpenMP().concurrency(),r.extent(0));

  Kokkos::Timer timer;
  // Reset duplicated array
  Kokkos::deep_copy(results,0);
  // Run loop only for OpenMP
  Kokkos::parallel_for("Duplicated Loop", 
    Kokkos::RangePolicy<Kokkos::OpenMP>(0,v.extent(0)),
    KOKKOS_LAMBDA(const int i) {
    // Every thread contribues to its version of the vector
    int tid = omp_get_thread_num();
    for(int j=0; j<v.extent(1); j++)
      results(tid,v(i,j))++;
  });
  // Contribute back to the single version
  Kokkos::parallel_for("Reduce Loop",
    Kokkos::RangePolicy<Kokkos::OpenMP>(0,v.extent(0)),
    KOKKOS_LAMBDA(const int i) {
    for(int tid=0; tid<results.extent(0); tid++)
      r(i)+=results(tid,i);
  });

  Kokkos::fence();
  double time = timer.seconds();
  return time;
}
#endif

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {

    int N = argc > 1?atoi(argv[1]):100000;
    int M = argc > 2?atoi(argv[2]):100;

    Kokkos::View<int**> values("V",N,M);
    Kokkos::View<int*> results("R",N);
    auto values_h = Kokkos::create_mirror_view(values);

    for(int i=0; i<N; i++)
      for(int j=0; j<M; j++)
	values_h(i,j) = rand()%N;

    Kokkos::deep_copy(values,values_h);

    double time_atomic = atomic_loop(values,results);
    std::cout << "Time Atomic: " << N << " " << M << " " << time_atomic << std::endl;
    #ifdef KOKKOS_ENABLE_OPENMP
    if(std::is_same<Kokkos::DefaultExecutionSpace,Kokkos::OpenMP>::value) {
      double time_dup = openmp_loop(values,results);
      std::cout << "Time Duplicated: " << N << " " << M << " " << time_dup << std::endl;
    }
    #endif

    double time_scatter_view = scatter_view_loop(values,results);
    std::cout << "Time ScatterView: " << N << " " << M << " " << time_scatter_view << std::endl;

  }
  Kokkos::finalize();
}
