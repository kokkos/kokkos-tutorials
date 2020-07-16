// ************************************************************************
// 
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
// 
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
// 
// ************************************************************************
//@HEADER

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
//#include <Kokkos_DualView.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <cstdlib>

// Kokkos provides two different random number generators with a 64 bit and a 1024 bit state.
// These generators are based on Vigna, Sebastiano (2014):
//           "An experimental exploration of Marsaglia's xorshift generators, scrambled"
//           See: http://arxiv.org/abs/1402.6246
// The generators can be used fully independently on each thread and have been tested to
// produce good statistics for both inter and intra thread numbers.
// Note that within a kernel NO random number operations are (team) collective operations.
// Everything can be called within branches. This is a difference from the curand library
// where certain operations are required to be called by all threads in a block.
//
// In Kokkos you are required to create a pool of generator states, so that threads can
// grep their own. On CPU architectures the pool size is equal to the thread number,
// on CUDA about 128k states are generated (enough to give every potentially simultaneously
// running thread its own state). With a kernel a thread is required to acquire a state from the
// pool and later return it.
// On CPUs the Random number generator is deterministic if using the same number of threads.
// On GPUs (i.e. using the CUDA backend) it is not deterministic because threads acquire states 
// via atomics.


//EXERCISE:
// 1. Create/Add a generator pool (suggest setting up a typedef for the pool)
// 2. Specify an instance of the generator pool to use in drawing random numbers from the pool
// 3. Provide constructor for generator pool and initialize data.
// 4. Use generator to draw random numbers from the pool as needed
// 5. Free generator (return to pool) for usage by other threads. 
// 6. Add (to main) a definition of the generator pool type and provide a random seed
//
// NB: the command line requires an integer, N, to define the number of "dart" throws (2^N)
// EXERCISE

// A Functor for generating uint64_t random numbers templated on the GeneratorPool type
//template<class RandPool=void>
struct GenRandom {

   // The GeneratorPool
   // RandPool rand_pool;

   typedef double Scalar;
   // typedef typename RandPool::generator_type gen_type;
           
   Scalar rad;   //target radius and box size
   long dart_groups; // Reuse the generator for drawing random #s this many times

   KOKKOS_INLINE_FUNCTION
   void operator() (long i, long& lsum) const {
     // Get a random number state from the pool for the active thread
     // gen_type rgen = rand_pool.get_state();

     // Draw samples numbers from the pool as urand64 between 0 and rand_pool.MAX_URAND64
     // Note there are function calls to get other type of scalars, and also to specify
     // Ranges or get a normal distributed float.
     for ( long it = 0; it < dart_groups; ++it ) {
     //  Scalar x = Kokkos::rand<gen_type, Scalar>::draw(rgen);
     //  Scalar y = Kokkos::rand<gen_type, Scalar>::draw(rgen);
        Scalar x = rand() / (double(RAND_MAX)+1) ;
        Scalar y = rand() / (double(RAND_MAX)+1) ;

     // Example - if you wish to draw from a normal distribution
     // mean = .5, stddev = 0.125
     //Scalar x = rgen.normal(.5,.125);
     //Scalar y = rgen.normal(.5,.125);

     Scalar dSq = (x*x + y*y);

     if (dSq <= rad*rad) { ++lsum; } // comparing to rad^2 - am I in the circle inscribed in square?
   }

   // Give the state back, which will allow another thread to aquire it
   // rand_pool.free_state(rgen);
 }  

 // Constructor, Initialize all members
 GenRandom( Scalar rad_, long dart_groups_) : rad(rad_), dart_groups(dart_groups_) {}

}; //end GenRandom struct


// Problem description:
// A 2-D quarter-space is assumed, x: 0 to s and y: 0 to s, where 's' is the side dimension.
// The sample space is thus a square; however a circular arc divides the space into an 'inside'
// the circle region and an 'outside' the circle. For a uniform sampling of the space, the 
// ratio of the number of 'hits' within (or on) the circle to the total 'hits' will define 
// the value of 'pi' according to the following formula:
//
//   # darts in the circular region/# darts in the square region :: 0.25*pi*side**2/side**2
//
//   which reduces to   pi = 4*# darts in circular region/#total darts thrown
//
// since the darts in the square region is just the total number of darts thrown.
// The sample parameters are the x,y position of the dart for each throw (sample).
//  
//       double x = rand();     double y = rand();
//
// and the circular region is defined by all x,y pairs within or on the circular arc.
// Using 'dist' as the distance from the center (0,0) of the sample space (the radius)
// 
//       double dist = sqrt(x*x + y*y);
//
// Hits are counted in the sample loop as follows:
//
//       if (dist <= side), hits-within-or-on-circle++;
//
// and pi is estimated at the completion of sampling as
//
//       double pi = 4 * hits-within-or-on-circle / darts
//
// Problem variations:
//  1) cycle on the sample size and compare pi vs sample size.
//  2) integer bit-size variation (64 vs 1024).


int main(int argc, char* args[]) {

  if ( argc < 2 ) {
    printf("RNG Example: Need at least one argument (number darts) to run; second optional argument for serial_iterations\n");
    return (-1);
  }

  Kokkos::initialize(argc,args);
  {
    const double rad = 1.0; // target radius (also box size)
    const long N = atoi(args[1]); // exponent used to create number of darts, 2^N

    const long dart_groups = argc > 2 ? atoi(args[2]) : 1 ;

    const long darts = std::pow(2,N);    // number of dart throws

    const double pi = 3.14159265358979323846 ;
    printf( "Reference Value for pi:  %lf\n",pi);

    // Create a random number generator pool (64-bit states or 1024-bit state)
    // Both take an 64 bit unsigned integer seed to initialize a Random_XorShift generator 
    // which is used to fill the generators of the pool.
    //typedef typename Kokkos::Random_XorShift64_Pool<> RandPoolType;
    //RandPoolType rand_pool(5374857);
    //Kokkos::Random_XorShift1024_Pool<> rand_pool1024(5374857);

    // Calcuate pi value for chosen sample size and generator type  
    long circHits = 0;
    //Kokkos::parallel_reduce ("MC-pi-estimate", darts/dart_groups, GenRandom<RandPoolType>(rad, dart_groups), circHits); 
    Kokkos::parallel_reduce ("MC-pi-estimate", darts/dart_groups, GenRandom(rad, dart_groups), circHits); 

    printf( "darts = %ld  hits = %ld  pi est = %lf\n", darts, circHits, 4.0*double(circHits)/double(darts) );
  }
  Kokkos::finalize();

  return 0;
}

