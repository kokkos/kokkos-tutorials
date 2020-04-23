#include<Kokkos_Core.hpp>
//EXERCISE: include the right header (later Kokkos will include this)
//#include<simd.hpp>

void test_simd(int N_in, int M, int R, double a) {

  //EXERCISE: get the right type here
  //using simd_t = ...;

  //EXERCISE: What will the N now be?
  int N = N_in;

  //EXERCISE: create SIMD Views instead
  Kokkos::View<double**> data("D",N,M);
  Kokkos::View<double*> results("R",N);

  // EXERCISE: create correctly a scalar view of results
  // For the final reduction we gonna need a scalar view of the data for now
  // Relying on knowing the data layout, we will add SIMD Layouts later
  // so that simple copy construction/assgnment would work
  Kokkos::View<double*> results_scalar(results);

  // Lets fill the data
  // EXERCISE: match the type of the scalar being copied in
  Kokkos::deep_copy(data,a);
  Kokkos::deep_copy(results,0.0);

  Kokkos::Timer timer;
  for(int r = 0; r<R; r++) {
    Kokkos::parallel_for("Combine",data.extent(0), KOKKOS_LAMBDA(const int i) {
      //EXERCISE Use the correct type here
      double tmp = 0.0;
      double b = a;
      for(int j=0; j<data.extent(1); j++) {
        tmp += b * data(i,j);
        b+=a+1.0*(j+1);
      }
      results(i) = tmp;
    });
    Kokkos::fence();
  }

  double time = timer.seconds();

  double value = 0.0;
  // Lets do the reduction here
  Kokkos::parallel_reduce("Reduce",results_scalar.extent(0), KOKKOS_LAMBDA(const int i, double& lsum) {
    lsum += results_scalar(i);
  },value);

  printf("SIMD Time: %lf ms ( %e )\n",time*1000,value);
}

void test_scalar(int N, int M, int R, double a) {

  Kokkos::View<double**> data("D",N,M);
  Kokkos::View<double*> results("R",N);

  Kokkos::deep_copy(data,a);
  Kokkos::deep_copy(results,0.0);

  Kokkos::Timer timer;
  for(int r = 0; r<R; r++) {
    Kokkos::parallel_for("Combine",data.extent(0), KOKKOS_LAMBDA(const int i) {
      double tmp = 0.0;
      double b = a;
      for(int j=0; j<data.extent(1); j++) {
        tmp += b * data(i,j);
        b+=a+1.0*(j+1);
      }
      results(i) = tmp;
    });
    Kokkos::fence();
  }

  double time = timer.seconds();

  double value = 0.0;
  Kokkos::parallel_reduce("Reduce",N, KOKKOS_LAMBDA(const int i, double& lsum) {
    lsum += results(i);
  },value);

  printf("Scalar Time: %lf ms ( %e )\n",time*1000,value);
}


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  
  int N = argc>1?atoi(argv[1]):320000;
  int M = argc>2?atoi(argv[2]):3;
  int R = argc>3?atoi(argv[3]):10;
  double scal = argc>4?atof(argv[4]):1.5;
  
  if(N%32) {
    printf("Please choose an N dividable by 32\n");
    return 0;
  }

  test_scalar(N,M,R,scal);
  test_simd(N,M,R,scal);

  Kokkos::finalize();
}
