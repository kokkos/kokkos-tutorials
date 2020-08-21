#include<Kokkos_Core.hpp>

struct System {
  // Using theoretical physicists way of describing system, 
  // i.e. we stick everything in as few constants as possible
  // be i and i+1 two timesteps dt apart: 
  // T(x,y,z)_(i+1) = T(x,y,z)_(i)+dT(x,y,z)*dt; 
  // dT(x,y,z) = q * sum_dxdydz( T(x+dx,y+dy,z+dz) - T(x,y,z) )
  // If its surface of the body add:
  // dT(x,y,z) += -sigma*T(x,y,z)^4
  // If its z==0 surface add incoming radiation energy
  // dT(x,y,0) += P

  // size of system
  int X,Y,Z;

  // number of timesteps
  int N;
  
  // interval for print
  int I;

  // Temperature and delta Temperature
  Kokkos::View<double***> T, dT;

  // Initial temperature
  double T0;

  // timestep width
  double dt;

  // thermal transfer coefficient 
  double q;

  // thermal radiation coefficient (assume Stefan Boltzmann law P = sigma*A*T^4
  double sigma;

  // incoming power
  double P;

  // init_system
  
  System() {
    X = 200;
    Y = 200;
    Z = 200;
    N = 10000;
    I = 100;
    T = Kokkos::View<double***>();
    dT = Kokkos::View<double***>();
    T0 = 0.0;
    dt = 0.1;
    q = 1.0;
    sigma = 1.0;
    P = 1.0;
  }

  void print_help() {
    printf("Options (default):\n");
    printf("  -X IARG: (%i) num elements in X direction\n", X); 
    printf("  -Y IARG: (%i) num elements in Y direction\n", Y); 
    printf("  -Z IARG: (%i) num elements in Z direction\n", Z); 
    printf("  -N IARG: (%i) num timesteps\n", N); 
    printf("  -I IARG: (%i) print interval\n", I); 
    printf("  -T0 FARG: (%lf) initial temperature\n", T0); 
    printf("  -dt FARG: (%lf) timestep size\n", dt); 
    printf("  -q FARG: (%lf) thermal conductivity\n", q); 
    printf("  -sigma FARG: (%lf) thermal radiation\n", sigma); 
    printf("  -P FARG: (%lf) incoming power\n", P);
  }

  // check command line args
  bool check_args(int argc, char* argv[]) {
    for(int i=1; i<argc; i++) {
      if(strcmp(argv[i],"-h")==0) { print_help(); return false; }
    }
    for(int i=1; i<argc; i++) {
      if(strcmp(argv[i],"-X")==0) X = atoi(argv[i+1]);
      if(strcmp(argv[i],"-Y")==0) Y = atoi(argv[i+1]);
      if(strcmp(argv[i],"-Z")==0) Z = atoi(argv[i+1]);
      if(strcmp(argv[i],"-N")==0) N = atoi(argv[i+1]);
      if(strcmp(argv[i],"-I")==0) I = atoi(argv[i+1]);
      if(strcmp(argv[i],"-T0")==0) T0 = atof(argv[i+1]);
      if(strcmp(argv[i],"-dt")==0) dt = atof(argv[i+1]);
      if(strcmp(argv[i],"-q")==0) q = atof(argv[i+1]);
      if(strcmp(argv[i],"-sigma")==0) sigma = atof(argv[i+1]);
      if(strcmp(argv[i],"-P")==0) P = atof(argv[i+1]);
    }
    T = Kokkos::View<double***>("System::T", X, Y, Z);
    dT = Kokkos::View<double***>("System::dT", X, Y, Z);
    return true;
  }
  // run_time_loops
  void timestep() {
    Kokkos::Timer timer;
    for(int t=0; t<=N; t++) {
      if(t>N/2) P = 0.0;
      compute_inner_dT();
      compute_surface_dT();
      double T_ave = compute_T();
      T_ave/=1e-9*(T.extent(0) * T.extent(1) * T.extent(2));
      }
      if(t%I == 0 || t==N) {
        double time = timer.seconds();
        printf("%i T=%lf Time (%lf %lf)\n",t,T_ave,time,time/t);
      }
    }
  }


  // Compute inner update
  struct ComputeInnerDT {};

  KOKKOS_FUNCTION
  void operator() (ComputeInnerDT, int x, int y, int z) const {
    double dT_xyz = 0.0;
    double T_xyz = T(x,y,z);
    dT_xyz += q * (T(x-1,y  ,z  ) - T_xyz);
    dT_xyz += q * (T(x+1,y  ,z  ) - T_xyz);
    dT_xyz += q * (T(x  ,y-1,z  ) - T_xyz);
    dT_xyz += q * (T(x  ,y+1,z  ) - T_xyz);
    dT_xyz += q * (T(x  ,y  ,z-1) - T_xyz);
    dT_xyz += q * (T(x  ,y  ,z+1) - T_xyz);

    dT(x,y,z) = dT_xyz;
  }
  void compute_inner_dT() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>,ComputeInnerDT>;
    int X = T.extent(0);
    int Y = T.extent(1);
    int Z = T.extent(2);
    Kokkos::parallel_for("ComputeInnerDT", policy_t({1,1,1},{X-1,Y-1,Z-1}), *this);
  };

  // Compute non-exposed surface
  // Dispatch makes sure that we don't hit elements twice
  enum {left,right,down,up,front,back};

  template<int Surface>
  struct ComputeSurfaceDT {};

  template<int Surface>
  KOKKOS_FUNCTION
  void operator() (ComputeSurfaceDT<Surface>, int i, int j) const {
    int X = T.extent(0);
    int Y = T.extent(1);
    int Z = T.extent(2);
    int x, y, z;
    if(Surface == left)  { x = 0;   y = i;   z = j; }
    if(Surface == right) { x = X-1; y = i;   z = j; }
    if(Surface == down)  { x = i;   y = 0;   z = j; }
    if(Surface == up)    { x = i;   y = Y-1; z = j; }
    if(Surface == front) { x = i;   y = j;   z = 0; }
    if(Surface == back)  { x = i;   y = j;   z = Z-1; }

    double dT_xyz = 0.0;
    double T_xyz = T(x,y,z);

    // Heat conduction to inner body
    if(x > 0)   dT_xyz += q * (T(x-1,y  ,z  ) - T_xyz);
    if(x < X-1) dT_xyz += q * (T(x+1,y  ,z  ) - T_xyz);
    if(y > 0)   dT_xyz += q * (T(x  ,y-1,z  ) - T_xyz);
    if(y < Y-1) dT_xyz += q * (T(x  ,y+1,z  ) - T_xyz);
    if(z > 0)   dT_xyz += q * (T(x  ,y  ,z-1) - T_xyz);
    if(z < Z-1) dT_xyz += q * (T(x  ,y  ,z+1) - T_xyz);

    // Incoming Power
    if(x == 0) dT_xyz += P;

    // thermal radiation
    int num_surfaces = ( x==0   ? 1 : 0)
                      +( x==X-1 ? 1 : 0)
                      +( y==0   ? 1 : 0)
                      +( y==Y-1 ? 1 : 0)
                      +( z==0   ? 1 : 0)
                      +( z==Z-1 ? 1 : 0);
    dT_xyz -= sigma * T_xyz * T_xyz * T_xyz * T_xyz * num_surfaces;
    dT(x,y,z) = dT_xyz;
  }

  void compute_surface_dT() {
    using policy_left_t =  Kokkos::MDRangePolicy<Kokkos::Rank<2>,ComputeSurfaceDT<left>>;
    using policy_right_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>,ComputeSurfaceDT<right>>;
    using policy_down_t =  Kokkos::MDRangePolicy<Kokkos::Rank<2>,ComputeSurfaceDT<down>>;
    using policy_up_t =    Kokkos::MDRangePolicy<Kokkos::Rank<2>,ComputeSurfaceDT<up>>;
    using policy_front_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>,ComputeSurfaceDT<front>>;
    using policy_back_t =  Kokkos::MDRangePolicy<Kokkos::Rank<2>,ComputeSurfaceDT<back>>;

    int X = T.extent(0);
    int Y = T.extent(1);
    int Z = T.extent(2);
    Kokkos::parallel_for("ComputeSurfaceDT_Left" , policy_left_t ({0,0},{Y,Z}),*this);
    Kokkos::parallel_for("ComputeSurfaceDT_Right", policy_right_t({0,0},{Y,Z}),*this);
    Kokkos::parallel_for("ComputeSurfaceDT_Down",  policy_down_t ({1,0},{X-1,Z}),*this);
    Kokkos::parallel_for("ComputeSurfaceDT_Up",    policy_up_t   ({1,0},{X-1,Z}),*this);
    Kokkos::parallel_for("ComputeSurfaceDT_front", policy_front_t({1,1},{X-1,Y-1}),*this);
    Kokkos::parallel_for("ComputeSurfaceDT_back",  policy_back_t ({1,1},{X-1,Y-1}),*this);
  }

  struct ComputeT {
    Kokkos::View<double***> T, dT;
    double dt;
    ComputeT(Kokkos::View<double***> T_, Kokkos::View<double***> dT_, double dt_):T(T_),dT(dT_),dt(dt_){}
    KOKKOS_FUNCTION
    void operator() (int x, int y, int z, double& sum_T) const {
      sum_T += T(x,y,z);
      T(x,y,z) += dt * dT(x,y,z);
    }
  };

  double compute_T() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>,Kokkos::IndexType<int>>;
    int X = T.extent(0);
    int Y = T.extent(1);
    int Z = T.extent(2);
    double sum_T;
    Kokkos::parallel_reduce("ComputeT", policy_t({0,0,0},{X,Y,Z}), ComputeT(T,dT,dt), sum_T);
    return sum_T;
  }
};

int main(int argc, char* argv[]) {
  //MPI_Init(argc,argv);
  Kokkos::initialize(argc,argv);
  {
    System sys;
    if(sys.check_args(argc,argv))
      sys.timestep();
  }

  Kokkos::finalize();
  //MPI_Finalize();
}
