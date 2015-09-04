/*
//@HEADER
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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
// 
// ************************************************************************
//@HEADER
*/

#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<sys/time.h>
#include<Kokkos_Core.hpp>

// Struct to keep all information about my MD system
struct Atoms {
  Kokkos::View<double*[3],Kokkos::LayoutRight> x;      //atom positions
  Kokkos::View<double*[3]> f;      //atom forces
  int natoms;     //number of atoms

  int neighbors_per_atom; // max neighbors per atom
  Kokkos::View<int*> num_neighbors;     // number of neighbors of each atom;
  Kokkos::View<int**> neighbors;         // neighbors of atoms

  double cut_force_2;
  double epsilon;
  double sigma6;

  int use_newton;
};

// Calculate the initial atom positions as a simple cubic lattice
// Create atoms in a way that a simple binned sorting is in place
// I.e. create atoms on a super grid of cells with each cell having
// also a simple cubic lattice
void create_sorted_atoms(Atoms& atoms, int n, int m) {
  //n is linear dimension of grid
  //m is linear dimension of cells in the grid
  atoms.natoms = n*n*n*m*m*m;
  atoms.x = Kokkos::View<double*[3],Kokkos::LayoutRight>("X",atoms.natoms);
  atoms.f = Kokkos::View<double*[3]>("F",atoms.natoms);

  atoms.num_neighbors = Kokkos::View<int*>("NumNeighs",atoms.natoms);

  //Loop over grid cells
  Kokkos::parallel_for("CreateAtoms", n*n*n, KOKKOS_LAMBDA (const int g) {
    const int g_x =  g/(n*n);
    const int g_y = (g%(n*n))/n;
    const int g_z =  g%n;

    for(int c_x=0; c_x<m; c_x++)
      for(int c_y=0; c_y<m; c_y++)
        for(int c_z=0; c_z<m; c_z++) {
          //calculate atom index
          const int i = ((g_x*n + g_y)*n + g_z) * m*m*m +
                        ((c_x*m + c_y)*m + c_z);
          //calculate positions
          atoms.x(i,0) = 1.0*(g_x*m + c_x);
          atoms.x(i,1) = 1.0*(g_y*m + c_y);
          atoms.x(i,2) = 1.0*(g_z*m + c_z);
    }
  });
}

// Create the neighbor list
size_t create_neighborlist(Atoms& atoms, double cut_neigh, int n, int m) {

  const double cut_neigh_2 = cut_neigh*cut_neigh;

  // How many gridcells need to be covered in each direction
  const int d_g = (cut_neigh+m-1)/m;

  // maximum number of neighbors per atom is
  // number of neighbor cells in cutoff times number of atoms in cell
  atoms.neighbors_per_atom = cut_neigh*cut_neigh*cut_neigh*5;

  atoms.neighbors = Kokkos::View<int**>("Neighbors",atoms.natoms,atoms.neighbors_per_atom);

  //Loop over grid cells
  Kokkos::parallel_for("CreateNeighbors", n*n*n, KOKKOS_LAMBDA (const int g) {
    const int g_x =  g/(n*n);
    const int g_y = (g%(n*n))/n;
    const int g_z =  g%n;

  for(int c_x=0; c_x<m; c_x++)
    for(int c_y=0; c_y<m; c_y++)
      for(int c_z=0; c_z<m; c_z++) {
        //calculate atom index
        const int i = ((g_x*n + g_y)*n + g_z) * m*m*m +
                      ((c_x*m + c_y)*m + c_z);
        //calculate positions
        const double x_i = atoms.x(i,0);
        const double y_i = atoms.x(i,1);
        const double z_i = atoms.x(i,2);

        atoms.num_neighbors[i] = 0;

    // Loop over neighbor cells
    for(int d_gx=-d_g; d_gx<=d_g; d_gx++) {
      if(g_x+d_gx>=0 && g_x+d_gx<n)
      for(int d_gy=-d_g; d_gy<=d_g; d_gy++) {
        if(g_y+d_gy>=0 && g_y+d_gy<n)
        for(int d_gz=-d_g; d_gz<=d_g; d_gz++) {
          if(g_z+d_gz>=0 && g_z+d_gz<n) {

            //calculate offset into neighbor cell
            int j_begin = (((g_x+d_gx)*n + (g_y+d_gy))*n + g_z+d_gz) * m*m*m;
            const int j_end = j_begin + m*m*m;

            //Skip Neighbor cells and atoms if we use newtons third law
            if(atoms.use_newton) {
              if(d_gx < 0) continue;
              else if(d_gx == 0 && d_gy <  0) continue;
              else if(d_gx == 0 && d_gy == 0 && d_gz <  0) continue;
              else if(d_gx == 0 && d_gy == 0 && d_gz == 0) j_begin = i+1;
            }

            //Loop over all atoms in that cell
            for(int j=j_begin; j<j_end; j++) {

              const double x_j = atoms.x(j,0);
              const double y_j = atoms.x(j,1);
              const double z_j = atoms.x(j,2);

              const double r2 = (x_j-x_i)*(x_j-x_i) + (y_j-y_i)*(y_j-y_i)
                              + (z_j-z_i)*(z_j-z_i);
              if(r2 < cut_neigh_2 && j!=i) {
                const int k = atoms.num_neighbors[i];
                atoms.num_neighbors[i]++;
                atoms.neighbors(i, k) = j;
              }
            } //end loop over atoms in neighbor cell
          }
        }
      }
    } //end loop over neighbor cells
  } //end loop over atoms in cell
  }); //end loop over grid

  // Compute number of interactions
  size_t n_interactions = 0;
  Kokkos::parallel_reduce("CountInteractions", atoms.natoms, KOKKOS_LAMBDA (const int& i, size_t& sum) {
    sum += atoms.num_neighbors[i];
  },n_interactions);

  return n_interactions;
}

template<int USE_NEWTON>
void compute_force(Atoms& atoms) {

  // Set force to zero
  Kokkos::parallel_for("ForceZero", atoms.natoms, KOKKOS_LAMBDA (const int& i) {
    atoms.f(i,0) = 0.0;
    atoms.f(i,1) = 0.0;
    atoms.f(i,2) = 0.0;
  });

  Kokkos::View<const double*[3],Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::RandomAccess> > x = atoms.x;
  Kokkos::View<double*[3],Kokkos::MemoryTraits<USE_NEWTON?Kokkos::Atomic:0> > f = atoms.f;

  // Loop over all atoms
  Kokkos::parallel_for("ComputeForce", atoms.natoms, KOKKOS_LAMBDA (const int& i) {

    // Use this to accumulate forces for atom i
    double f_x = 0.0;
    double f_y = 0.0;
    double f_z = 0.0;

    // Load position of atom i
    const double x_i = x(i,0);
    const double y_i = x(i,1);
    const double z_i = x(i,2);

    // Load number of neighbors of atom i
    const int num_j = atoms.num_neighbors[i];

    // Loop over neighbors j of atom i
    for(int jj = 0; jj<num_j; jj++) {
      // Load the neighbor index
      const int j = atoms.neighbors(i,jj);

      // Load neighbor positions
      const double x_j = x(j,0);
      const double y_j = x(j,1);
      const double z_j = x(j,2);

      // Calculate distance vector
      const double d_x = x_j-x_i;
      const double d_y = y_j-y_i;
      const double d_z = z_j-z_i;

      // Calculate distance square
      const double r2 = d_x*d_x + d_y*d_y + d_z*d_z;

      // If distance smaller than cutoff compute force for atoms i-j
      if(r2 < atoms.cut_force_2) {
        const double r2inv = 1.0/r2;
        const double sigma6_r6inv = atoms.sigma6*r2inv*r2inv*r2inv;

        const double force = -24.0 * atoms.epsilon *
          (2. * sigma6_r6inv * sigma6_r6inv - sigma6_r6inv) * r2inv;

        // Accumulate force for atom i
        f_x += force * d_x;
        f_y += force * d_y;
        f_z += force * d_z;

        // If we are using newtons third law add force to atom j
        if(USE_NEWTON) {
          f(j,0) -= force * d_x;
          f(j,1) -= force * d_y;
          f(j,2) -= force * d_z;
        }
      }
    }
    // Add force to atom i
    f(i,0) += f_x;
    f(i,1) += f_y;
    f(i,2) += f_z;
  });
}

int main(int argc, char* argv[]) {
  // Set default values for system
  int gx = 20; // linear number of cells in box (gx*gx*gx*cx*cx*cx is number of atoms)
  int cx = 3;  // linear number of atoms per cell
  double cut_force = 2.5; // force cutoff
  double cut_neigh = 3.0; // neighbor list cutoff
  int use_newton = 1;     // use Newton's third law
  int nrepeat = 10;       // number of force kernel invocations

  // Read command line arguments
  for(int i=0; i<argc; i++) {
           if( strcmp(argv[i], "-gx") == 0) {
      gx = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-cx") == 0) {
      cx = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-cut_force") == 0) {
      cut_force = atof(argv[++i]);
    } else if( strcmp(argv[i], "-cut_neigh") == 0) {
      cut_neigh = atof(argv[++i]);
    } else if( strcmp(argv[i], "-use_newton") == 0) {
      use_newton = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("MDForceKernel Otpions:\n");
      printf("  -gx <int>:          set the linear number of cells in box (default 20)\n ");
      printf("                      gx*gx*gx*cx*cx*cx is number of atoms\n");
      printf("  -cx <int>:          set the linear number of atoms per cell (default 3)\n");
      printf("                      gx*gx*gx*cx*cx*cx is number of atoms\n");
      printf("  -cut_force <float>: set force cutoff (default 2.5)\n");
      printf("  -cut_neigh <float>: set neighbor cutoff (default 3.0)\n");
      printf("  -use_newton <int>:  use Newton's third law (default 1)\n");
      printf("  -nrepeat <int>:     number of force kernel invocations (default 10)\n");
      printf("  -help (-h):         print this message\n");
    }
  }

  // Set up system
  Atoms atoms;
  atoms.use_newton = use_newton;
  atoms.cut_force_2 = cut_force*cut_force;
  atoms.epsilon = 1;
  atoms.sigma6 = 1;

  //Initialize Kokkos
  Kokkos::initialize(argc,argv);

  // Create the atoms
  create_sorted_atoms(atoms,gx,cx);

  // Create the neighborlist
  size_t n_interactions = create_neighborlist(atoms,cut_neigh,gx,cx);

  // Time force computation
  struct timeval begin,end;

  gettimeofday(&begin,NULL);

  for(int t=0; t<nrepeat; t++)
    if(use_newton)
      compute_force<1>(atoms);
    else
      compute_force<0>(atoms);

  gettimeofday(&end,NULL);

  // Print results
  double time = 1.0*(end.tv_sec-begin.tv_sec) + 1.0e-6*(end.tv_usec-begin.tv_usec);
  printf("natoms time cut_force cut_neigh ave_neighbors use_newton interactions_per_sec FOM: \n");
  printf("%i %lf %lf %lf %i %lf %lf %lf\n",atoms.natoms,time,cut_force,cut_neigh,1.0*n_interactions/atoms.natoms,use_newton,
                                   1.0e-6*n_interactions*nrepeat/time,1.0e-6*atoms.natoms*nrepeat/time);

  //Finalize Kokkos
  Kokkos::finalize();

}



