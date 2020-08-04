#include<Kokkos_Core.hpp>

using atomic_1d_view = Kokkos::View<int*, Kokkos::DefaultExecutionSpace, 
                                           Kokkos::MemoryTraits<Kokkos::Atomic> >;
using atomic_2d_view = Kokkos::View<int**, Kokkos::DefaultExecutionSpace, 
                                           Kokkos::MemoryTraits<Kokkos::Atomic> >;
// Scatter Add algorithm using data replication
double scatter_add_loop(Kokkos::View<int**> v, 
		 atomic_1d_view r) {
  Kokkos::Timer timer;
  Kokkos::Experimental::UniqueToken<Kokkos::DefaultExecutionSpace> tokens;
  int dup_size = tokens.size() > 2048 ? 2048 : tokens.size();
  int dup_mask = 2048 - 1;
  atomic_2d_view results("Rdup",dup_size,r.extent(0));

  // Reset duplicated array
  Kokkos::deep_copy(results,0);

  Kokkos::parallel_for("Duplicated Loop", 
    Kokkos::RangePolicy<>(0,v.extent(0)),
    KOKKOS_LAMBDA(const int i) {

    // Every thread contribues to its version of the vector
    Kokkos::Experimental::AcquireUniqueToken<Kokkos::DefaultExecutionSpace> token_val(
        tokens);
    const int32_t tid = token_val.value() & dup_mask;
    for(int j=0; j<v.extent(1); j++)
      results(tid,v(i,j))++;
  });

  // Contribute back to the single version
  Kokkos::parallel_for("Reduce Loop",
    Kokkos::TeamPolicy<>(v.extent(0), Kokkos::AUTO),
    KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team) {
       int i = team.league_rank();
       Kokkos::parallel_for( Kokkos::TeamVectorRange(team, 
                                results.extent(0)), [=] (int tid) {
          r(i)+=results(tid,i);
       });
  });

  Kokkos::fence();
  double time = timer.seconds();
  return time;
}

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

    double time_dup = scatter_add_loop(values,results);
    std::cout << "Time Duplicated: " << N << " " << M << " " << time_dup << std::endl;

  }
  Kokkos::finalize();
}
