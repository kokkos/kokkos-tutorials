#include<Kokkos_Core.hpp>

// EXERCISE: need to remove the ifdef...
#ifdef KOKKOS_ENABLE_OPENMP

using atomic_2d_view = Kokkos::View<int**, Kokkos::DefaultExecutionSpace, 
                                           Kokkos::MemoryTraits<Kokkos::Atomic> >;

// Scatter Add algorithm using data replication
double scatter_add_loop(Kokkos::View<int**> v, 
		 Kokkos::View<int*> r, int dup_size) {
  Kokkos::Timer timer;

  // need a mask below so that the thread ids can be mapped to the results view in the event
  // there are more threads than dup_size   
  int dup_mask = dup_size - 1;  

  // EXERCISE: need to create an instance of UniqueToken that is the same size or bigger
  //           than the number of teams below

  // accumulator for each thread
  // dup_size is used so that the view will fit in memory.
  atomic_2d_view results("Rdup",dup_size,r.extent(0));

  // Reset duplicated array
  Kokkos::deep_copy(results,0);

  using team_policy_type = Kokkos::TeamPolicy<>;
  team_policy_type team_policy(v.extent(0), Kokkos::AUTO);

  Kokkos::parallel_for("Accumulator Loop", 
    team_policy, KOKKOS_LAMBDA(team_policy_type::member_type team) {
    int i = team.league_rank();

    // EXERCISE: replace omp_get_thread_num() with token acquire
    // HINT, Token is per team so acquire should be single and broadcast
    int tid = omp_get_thread_num() & dup_mask;

    Kokkos::parallel_for( Kokkos::TeamVectorRange(team, v.extent(1)), [=] (int j) {
       results(tid,v(i,j))++;
    });

    // EXERCISE: don't forget to release token
  });
  
  // Contribute back to the single version
  Kokkos::parallel_for("Reduce Loop",
    Kokkos::TeamPolicy<>(r.extent(0), Kokkos::AUTO),
    KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team) {
       int i = team.league_rank();
       int r_local = 0;
       Kokkos::parallel_reduce( Kokkos::TeamVectorRange(team, 
                                results.extent(0)), [=] (int tid, int & r_) {
          r_ += results(tid,i);
       }, r_local);

       Kokkos::single(Kokkos::PerTeam(team), [&] () {
          r(i) = r_local;
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
 
    // EXERCISE: D has to be such that the results view above will
    //           fit into memory.  you can't use concurrency with 
    //           HIP and CUDA because it is too big.
    // note that the third parameter is a pow(2)
    int D = argc > 3?1<<atoi(argv[3]):
        Kokkos::OpenMP::concurrency();

    Kokkos::View<int**> values("V",N,M);
    Kokkos::View<int*> results("R",N);
    auto values_h = Kokkos::create_mirror_view(values);

    for(int i=0; i<N; i++)
      for(int j=0; j<M; j++)
	values_h(i,j) = rand()%N;

    Kokkos::deep_copy(values,values_h);

    double time_dup = scatter_add_loop(values,results,D);
    std::cout << "Time Duplicated: " << N << " " << M << " " << time_dup << std::endl;

  }
  Kokkos::finalize();
}

#endif
