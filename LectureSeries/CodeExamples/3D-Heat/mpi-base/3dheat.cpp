#include<Kokkos_Core.hpp>
#include<mpi.h>

struct CommHelper {
  MPI_Comm comm;

  // Num MPI ranks in each dimension
  int nx, ny, nz;

  // My rank
  int me;

  // My pos in proc grid
  int x, y, z;

  // Neighbor Ranks
  int up,down,left,right,front,back;
 
  CommHelper(MPI_Comm comm_) {
    comm = comm_;
    int nranks;
    MPI_Comm_size(comm, &nranks);
    MPI_Comm_rank(comm, &me);

    nx = std::pow(1.0*nranks,1.0/3.0);
    while(nranks%nx != 0) nx++;

    ny = std::sqrt(1.0*(nranks/nx));
    while((nranks/nx)%ny != 0) ny++;
    
    nz = nranks/nx/ny;
    x = me%nx;
    y = (me/nx)%ny;
    z = (me/nx/ny);
    left  = x==0 ?-1:me-1;
    right = x==nx-1?-1:me+1;
    down  = y==0 ?-1:me-nx;
    up    = y==ny-1?-1:me+nx;
    front = z==0 ?-1:me-nx*ny;
    back  = z==nz-1?-1:me+nx*ny;

    printf("NumRanks: %i Me: %i Grid: %i %i %i MyPos: %i %i %i\n",nranks,me,nx,ny,nz,x,y,z);
    printf("Me: %i MyNeighs: %i %i %i %i %i %i\n",me,left,right,down,up,front,back);
  }

  template<class ViewType>
  void isend_irecv(int partner, ViewType send_buffer, ViewType recv_buffer, MPI_Request* request_send, MPI_Request* request_recv) {
    MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_recv);
    MPI_Isend(send_buffer.data(), send_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_send);
  }
};

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

  // Communicator
  CommHelper comm;
  MPI_Request mpi_requests_recv[6];
  MPI_Request mpi_requests_send[6];
  int mpi_active_requests;

  // size of system
  int X,Y,Z;

  // Local box
  int X_lo, Y_lo, Z_lo;
  int X_hi, Y_hi, Z_hi;

  // number of timesteps
  int N;
  
  // interval for print
  int I;

  // Temperature and delta Temperature
  Kokkos::View<double***> T, dT;
  // Halo data
  Kokkos::View<double**> T_left, T_right, T_up, T_down, T_front, T_back;
  Kokkos::View<double**> T_left_out, T_right_out, T_up_out, T_down_out, T_front_out, T_back_out;

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
  
  System(MPI_Comm comm_): comm(comm_) {
    mpi_active_requests = 0;
    X = Y = Z = 200;
    X_lo = Y_lo = Z_lo = 0;
    X_hi = Y_hi = Z_hi = X; 
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

  void setup_subdomain() {
    int dX = (X+comm.nx-1)/comm.nx;
    X_lo = dX*comm.x;
    X_hi = X_lo + dX;
    if(X_hi>X) X_hi=X;
    int dY = (Y+comm.ny-1)/comm.ny;
    Y_lo = dY*comm.y;
    Y_hi = Y_lo + dY;
    if(Y_hi>Y) Y_hi=Y;
    int dZ = (Z+comm.nz-1)/comm.nz;
    Z_lo = dZ*comm.z;
    Z_hi = Z_lo + dZ;
    if(Z_hi>Z) Z_hi=X;

    printf("My Domain: %i (%i %i %i) (%i %i %i)\n",comm.me,X_lo,Y_lo,Z_lo,X_hi,Y_hi,Z_hi);
    T = Kokkos::View<double***>("System::T", X_hi - X_lo, Y_hi - Y_lo, Z_hi - Z_lo);
    dT = Kokkos::View<double***>("System::dT", T.extent(0), T.extent(1), T.extent(2));
    // incoming halos
    if(X_lo != 0) T_left  = Kokkos::View<double**>("System::T_left" , Y_hi - Y_lo, Z_hi - Z_lo);
    if(X_hi != X) T_right = Kokkos::View<double**>("System::T_right", Y_hi - Y_lo, Z_hi - Z_lo);
    if(Y_lo != 0) T_up    = Kokkos::View<double**>("System::T_up"   , X_hi - X_lo, Z_hi - Z_lo);
    if(Y_hi != Y) T_down  = Kokkos::View<double**>("System::T_down" , X_hi - X_lo, Z_hi - Z_lo);
    if(Z_lo != 0) T_front = Kokkos::View<double**>("System::T_front", X_hi - X_lo, Y_hi - Y_lo);
    if(Z_hi != Z) T_back  = Kokkos::View<double**>("System::T_back" , X_hi - X_lo, Y_hi - Y_lo);

    // outgoing halo
    if(X_lo != 0) T_left_out  = Kokkos::View<double**>("System::T_left_out" , Y_hi - Y_lo, Z_hi - Z_lo);
    if(X_hi != X) T_right_out = Kokkos::View<double**>("System::T_right_out", Y_hi - Y_lo, Z_hi - Z_lo);
    if(Y_lo != 0) T_up_out    = Kokkos::View<double**>("System::T_up_out"   , X_hi - X_lo, Z_hi - Z_lo);
    if(Y_hi != Y) T_down_out  = Kokkos::View<double**>("System::T_down_out" , X_hi - X_lo, Z_hi - Z_lo);
    if(Z_lo != 0) T_front_out = Kokkos::View<double**>("System::T_front_out", X_hi - X_lo, Y_hi - Y_lo);
    if(Z_hi != Z) T_back_out  = Kokkos::View<double**>("System::T_back_out" , X_hi - X_lo, Y_hi - Y_lo);
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
    setup_subdomain();
    return true;
  }

  // run_time_loops
  void timestep() {
    Kokkos::Timer timer;
    for(int t=0; t<=N; t++) {
      if(t>N/2) P = 0.0;
      exchange_T_halo();
      compute_inner_dT();
      compute_surface_dT();
      double T_ave = compute_T();
      T_ave/=1e-9*(X * Y * Z);
      if((t%I == 0 || t==N) && (comm.me==0)) {
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
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>,ComputeInnerDT,int>;
    int myX = T.extent(0);
    int myY = T.extent(1);
    int myZ = T.extent(2);
    Kokkos::parallel_for("ComputeInnerDT", policy_t({1,1,1},{myX-1,myY-1,myZ-1}), *this);
  };

  // Compute non-exposed surface
  // Dispatch makes sure that we don't hit elements twice
  enum {left,right,down,up,front,back};

  template<int Surface>
  struct ComputeSurfaceDT {};

  template<int Surface>
  KOKKOS_FUNCTION
  void operator() (ComputeSurfaceDT<Surface>, int i, int j) const {
    int NX = T.extent(0);
    int NY = T.extent(1);
    int NZ = T.extent(2);
    int x, y, z;
    if(Surface == left)  { x = 0;    y = i;    z = j; }
    if(Surface == right) { x = NX-1; y = i;    z = j; }
    if(Surface == down)  { x = i;    y = 0;    z = j; }
    if(Surface == up)    { x = i;    y = NY-1; z = j; }
    if(Surface == front) { x = i;    y = j;    z = 0; }
    if(Surface == back)  { x = i;    y = j;    z = NZ-1; }

    double dT_xyz = 0.0;
    double T_xyz = T(x,y,z);

    // Heat conduction to inner body
    if(x > 0)    dT_xyz += q * (T(x-1,y  ,z  ) - T_xyz);
    if(x < NX-1) dT_xyz += q * (T(x+1,y  ,z  ) - T_xyz);
    if(y > 0)    dT_xyz += q * (T(x  ,y-1,z  ) - T_xyz);
    if(y < NY-1) dT_xyz += q * (T(x  ,y+1,z  ) - T_xyz);
    if(z > 0)    dT_xyz += q * (T(x  ,y  ,z-1) - T_xyz);
    if(z < NZ-1) dT_xyz += q * (T(x  ,y  ,z+1) - T_xyz);

    // Heat conduction with Halo    
    if(x == 0 && X_lo != 0)  dT_xyz += q * (T_left(y  ,z  ) - T_xyz);
    if(x == (NX-1) && X_hi != X)  dT_xyz += q * (T_right(y  ,z  ) - T_xyz);
    if(y == 0 && Y_lo != 0)  dT_xyz += q * (T_down(x  ,z  ) - T_xyz);
    if(y == (NY-1) && Y_hi != Y)  dT_xyz += q * (T_up(x  ,z  ) - T_xyz);
    if(z == 0 && Z_lo != 0)  dT_xyz += q * (T_front(x  ,y  ) - T_xyz);
    if(z == (NZ-1) && Z_hi != Z)  dT_xyz += q * (T_back(x  ,y  ) - T_xyz);

    // Incoming Power
    if(x == 0 && X_lo == 0) dT_xyz += P;

    // thermal radiation
    int num_surfaces = ( (x==0  && X_lo == 0) ? 1 : 0)
                      +( (x==(NX-1) && X_hi == X) ? 1 : 0)
                      +( (y==0  && Y_lo == 0) ? 1 : 0)
                      +( (y==(NY-1) && Y_hi == Y) ? 1 : 0)
                      +( (z==0  && Z_lo == 0) ? 1 : 0)
                      +( (z==(NZ-1) && Z_hi == Z) ? 1 : 0);
    dT_xyz -= sigma * T_xyz * T_xyz * T_xyz * T_xyz * num_surfaces;
    dT(x,y,z) = dT_xyz;
  }

  void exchange_T_halo() {
    mpi_active_requests = 0;
    int mar = 0;
    if(X_lo != 0) {
      Kokkos::deep_copy(T_left_out ,Kokkos::subview(T,0,Kokkos::ALL,Kokkos::ALL));
      comm.isend_irecv(comm.left,T_left_out,T_left,&mpi_requests_send[mar],&mpi_requests_recv[mar]);
      mar++;
    }
    if(Y_lo != 0) {
      Kokkos::deep_copy(T_down_out ,Kokkos::subview(T,Kokkos::ALL,0,Kokkos::ALL));
      comm.isend_irecv(comm.down,T_down_out,T_down,&mpi_requests_send[mar],&mpi_requests_recv[mar]);
      mar++;
    }
    if(Z_lo != 0) { 
      Kokkos::deep_copy(T_front_out,Kokkos::subview(T,Kokkos::ALL,Kokkos::ALL,0));
      comm.isend_irecv(comm.front,T_front_out,T_front,&mpi_requests_send[mar],&mpi_requests_recv[mar]);
      mar++;
    }
    if(X_hi != X) {
      Kokkos::deep_copy(T_right_out,Kokkos::subview(T,X_hi-X_lo-1,Kokkos::ALL,Kokkos::ALL));
      comm.isend_irecv(comm.right,T_right_out,T_right,&mpi_requests_send[mar],&mpi_requests_recv[mar]);
      mar++;
    }
    if(Y_hi != Y) {
      Kokkos::deep_copy(T_up_out,   Kokkos::subview(T,Kokkos::ALL,Y_hi-Y_lo-1,Kokkos::ALL));
      comm.isend_irecv(comm.up,T_up_out,T_up,&mpi_requests_send[mar],&mpi_requests_recv[mar]);
      mar++;
    }
    if(Z_hi != Z) {
      Kokkos::deep_copy(T_back_out, Kokkos::subview(T,Kokkos::ALL,Kokkos::ALL,Z_hi-Z_lo-1));
      comm.isend_irecv(comm.back,T_back_out,T_back,&mpi_requests_send[mar],&mpi_requests_recv[mar]);
      mar++;
    }
    mpi_active_requests = mar;
  }

  void compute_surface_dT() {
    using policy_left_t =  Kokkos::MDRangePolicy<Kokkos::Rank<2>,ComputeSurfaceDT<left>,int>;
    using policy_right_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>,ComputeSurfaceDT<right>,int>;
    using policy_down_t =  Kokkos::MDRangePolicy<Kokkos::Rank<2>,ComputeSurfaceDT<down>,int>;
    using policy_up_t =    Kokkos::MDRangePolicy<Kokkos::Rank<2>,ComputeSurfaceDT<up>,int>;
    using policy_front_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>,ComputeSurfaceDT<front>,int>;
    using policy_back_t =  Kokkos::MDRangePolicy<Kokkos::Rank<2>,ComputeSurfaceDT<back>,int>;

    int X = T.extent(0);
    int Y = T.extent(1);
    int Z = T.extent(2);
    if(mpi_active_requests>0) {
      MPI_Waitall(mpi_active_requests,mpi_requests_send,MPI_STATUSES_IGNORE);
      MPI_Waitall(mpi_active_requests,mpi_requests_recv,MPI_STATUSES_IGNORE);
    }
    Kokkos::parallel_for("ComputeSurfaceDT_Left" , policy_left_t ({0,0},{Y,Z}),*this);
    Kokkos::parallel_for("ComputeSurfaceDT_Right", policy_right_t({0,0},{Y,Z}),*this);
    Kokkos::parallel_for("ComputeSurfaceDT_Down",  policy_down_t ({1,0},{X-1,Z}),*this);
    Kokkos::parallel_for("ComputeSurfaceDT_Up",    policy_up_t   ({1,0},{X-1,Z}),*this);
    Kokkos::parallel_for("ComputeSurfaceDT_front", policy_front_t({1,1},{X-1,Y-1}),*this);
    Kokkos::parallel_for("ComputeSurfaceDT_back",  policy_back_t ({1,1},{X-1,Y-1}),*this);
  }

  // Some compilers have deduction issues if this were just a tagged operator
  // So did a full Functor here instead
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
    double my_T;
    Kokkos::parallel_reduce("ComputeT", policy_t({0,0,0},{X,Y,Z}), ComputeT(T,dT,dt), my_T);
    double sum_T;
    MPI_Allreduce(&my_T,&sum_T,1,MPI_DOUBLE,MPI_SUM,comm.comm);
    return sum_T;
  }
};

int main(int argc, char* argv[]) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize(argc,argv);
  {
    System sys(MPI_COMM_WORLD);
    if(sys.check_args(argc,argv))
      sys.timestep();
  }

  Kokkos::finalize();
  MPI_Finalize();
}
