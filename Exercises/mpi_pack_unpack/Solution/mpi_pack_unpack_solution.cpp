// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// EXERCISE Goal:
// Understand how to use MPI with Kokkos.

#include<Kokkos_Core.hpp>
#include<mpi.h>

struct TagPack{};
struct TagUnpack{};
template<class DataMemorySpace, class BufferMemorySpace>
struct RunPackCommUnpackTest {

  // Lists Of Things to Send 
  Kokkos::View<int*,DataMemorySpace> list;
  Kokkos::View<int*,Kokkos::HostSpace> list_h;

  // Data 
  Kokkos::View<double*,DataMemorySpace> data;

  // Send and receive buffers for device and host
  Kokkos::View<double*,BufferMemorySpace> send_buffer;
  typename Kokkos::View<double*,BufferMemorySpace>::host_mirror_type send_buffer_h;
  Kokkos::View<double*,BufferMemorySpace> recv_buffer;
  typename Kokkos::View<double*,BufferMemorySpace>::host_mirror_type recv_buffer_h;

  // ExecutionSpace to run the pack/unpack kernel on
  using exec_space = typename DataMemorySpace::execution_space;

  // Should we give device buffers to MPI?
  bool use_device_buffer;  

  int me,partner;

  RunPackCommUnpackTest(int N, int B, bool use_device_buffer_) {
    // Just creating all of the arrays
    list    = Kokkos::View<int*,DataMemorySpace>("List",B);
    list_h  = Kokkos::View<int*,Kokkos::HostSpace>("List_h",B);
    data    = Kokkos::View<double*,DataMemorySpace>("Data",N);
    send_buffer   = Kokkos::View<double*,BufferMemorySpace>("SendBuffer",B);
    send_buffer_h = Kokkos::create_mirror_view(send_buffer);
    recv_buffer   = Kokkos::View<double*,BufferMemorySpace>("RecvBuffer",B);
    recv_buffer_h = Kokkos::create_mirror_view(recv_buffer);
    use_device_buffer = use_device_buffer_;

    for(int i=0; i<B; i++)
      list_h(i) = rand()%N;
    Kokkos::deep_copy(list,list_h);
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    if(me==0) partner=1; else partner=0;
  }

  void run_comm() {
    // Get the raw pointer for the MPI copy depending on use_device_buffer
    void* recv_buf = use_device_buffer?recv_buffer.data():recv_buffer_h.data();
    void* send_buf = use_device_buffer?send_buffer.data():send_buffer_h.data();

    // Post the Receives, create requests and MPI_Irecv into recv_buf
    MPI_Request requests[2];
    MPI_Irecv(recv_buf,list.extent(0),MPI_DOUBLE,partner,1,MPI_COMM_WORLD,&requests[0]);

    // Pack the buffer
    Kokkos::parallel_for(Kokkos::RangePolicy<exec_space,TagPack>(0,list.extent(0)),*this);
    // deep copy device send buffer to host send buffer if necessary (depending
    if(!use_device_buffer) Kokkos::deep_copy(send_buffer_h,send_buffer);
    else Kokkos::fence();
 
    // Send send_buffer through MPI_Send
    MPI_Isend(send_buf,list.extent(0),MPI_DOUBLE,partner,1,MPI_COMM_WORLD,&requests[1]);

    // Wait for the requests to finish using MPI_Waitall
    MPI_Waitall(2,requests,MPI_STATUSES_IGNORE);

    // deep copy host receive buffer to device if necessary (depending on use_device_buffer)
    if(!use_device_buffer) Kokkos::deep_copy(recv_buffer,recv_buffer_h);

    // Unpack the buffer
    Kokkos::parallel_for(Kokkos::RangePolicy<exec_space,TagUnpack>(0,list.extent(0)),*this);
  }

  // Pack and Unpack Operators
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagPack, const int i) const {
    send_buffer(i) = data(list(i));
  }
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagUnpack, const int i) const {
    data(list(i)) = recv_buffer(i);
  }

};

template<class DataMemorySpace, class BufferMemorySpace>
void run(int N, int B, int R, bool use_device_buffer) {
     RunPackCommUnpackTest<DataMemorySpace,BufferMemorySpace> comm(N,B,use_device_buffer);
     MPI_Barrier(MPI_COMM_WORLD);
     Kokkos::Timer timer;
     for(int r=0; r<R; r++) {
       comm.run_comm();
       Kokkos::fence();
     }
     MPI_Barrier(MPI_COMM_WORLD);
     double time=timer.seconds();
     printf("Data: %s DeviceBuffer: %s Time: %lf CopyToHostForMPI: %s\n",
       DataMemorySpace::name(),BufferMemorySpace::name(),time,
       use_device_buffer?"no":"yes");
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize(argc,argv);

  // Size of Data
  int N = argc>1?atoi(argv[1]):10000000;

  // Size of Buffer
  int B = argc>2?atoi(argv[2]):100000;

  // Number of repeats for timing
  int R = argc>3?atoi(argv[3]):10;

  // Copy to host first?
  bool use_device_buffer = argc>4? atoi(argv[4])==1:true;

  {
    #ifdef KOKKOS_ENABLE_CUDA
    run<Kokkos::CudaSpace,Kokkos::CudaSpace>(N,B,R,use_device_buffer);
    run<Kokkos::CudaSpace,Kokkos::CudaHostPinnedSpace>(N,B,R,use_device_buffer);
    run<Kokkos::CudaUVMSpace,Kokkos::CudaUVMSpace>(N,B,R,use_device_buffer);
    #endif
    #ifdef KOKKOS_ENABLE_HIP
    run<Kokkos::HIPSpace,Kokkos::HIPSpace>(N,B,R,use_device_buffer);
    run<Kokkos::HIPSpace,Kokkos::HIPHostPinnedSpace>(N,B,R,use_device_buffer);
    #endif
    run<Kokkos::HostSpace,Kokkos::HostSpace>(N,B,R,use_device_buffer);
  }
  Kokkos::finalize();
  MPI_Finalize();
}
