#include<Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_OPENMP)
// Scatter Add algorithm using data replication
double scatter_add_loop(Kokkos::View<int**,Kokkos::HostSpace> v, 
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

    Kokkos::deep_copy(values_h,values);

    #ifdef KOKKOS_ENABLE_OPENMP
    if(std::is_same<Kokkos::DefaultExecutionSpace,Kokkos::OpenMP>::value) {
      double time_dup = openmp_loop(values,results);
      std::cout << "Time Duplicated: " << N << " " << M << " " << time_dup << std::endl;
    }
    #endif

  }
  Kokkos::finalize();
}
