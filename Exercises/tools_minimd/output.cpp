/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "mpi.h"
#include "ljs.h"
#include "atom.h"
#include "integrate.h"
#include "force.h"
#include "neighbor.h"
#include "comm.h"
#include "thermo.h"
#include "timer.h"
#include <time.h>
#include "variant.h"

void stats(int, double*, double*, double*, double*, int, int*);

void output(In &in, Atom &atom, Force* force, Neighbor &neighbor, Comm &comm,
            Thermo &thermo, Integrate &integrate, Timer &timer, int screen_yaml)
{
  int i, n;
  int histo[10];
  double tmp, ave, max, min, total;
  FILE* fp;

  const int me = comm.me;
  const int nprocs = comm.nprocs;
  const int nthreads = Kokkos::HostSpace::execution_space().concurrency();


  /* enforce PBC, then check for lost atoms */

  atom.pbc();

  int natoms;
  MPI_Allreduce(&atom.nlocal, &natoms, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int nlost = 0;

  for(i = 0; i < atom.nlocal; i++) {
    if(atom.x(i,0) < 0.0 || atom.x(i,0) >= atom.box.xprd ||
        atom.x(i,1) < 0.0 || atom.x(i,1) >= atom.box.yprd ||
        atom.x(i,2) < 0.0 || atom.x(i,2) >= atom.box.zprd) nlost++;
  }

  int nlostall;
  MPI_Allreduce(&nlost, &nlostall, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(natoms != atom.natoms || nlostall > 0) {
    if(me == 0) printf("Atom counts = %d %d %d\n",
                         nlostall, natoms, atom.natoms);

    if(me == 0) printf("ERROR: Incorrect number of atoms\n");

    return;
  }

  /* long-range energy and pressure corrections Whats this???*/

  double engcorr = 8.0 * 3.1415926 * in.rho *
                   (1.0 / (9.0 * pow(force->cutforce, MMD_float(9.0))) - 1.0 / (3.0 * pow(force->cutforce, MMD_float(3.0))));
  double prscorr = 8.0 * 3.1415926 * in.rho * in.rho *
                   (4.0 / (9.0 * pow(force->cutforce, MMD_float(9.0))) - 2.0 / (3.0 * pow(force->cutforce, MMD_float(3.0))));

  /* thermo output */

  double conserve;

  if(me == 0) {


    time_t general_time = time(NULL);
    struct tm local_time = *localtime(&general_time);
    char filename[256];

    sprintf(filename, "miniMD-%4d-%02d-%02d-%02d-%02d-%02d.yaml",
            local_time.tm_year + 1900, local_time.tm_mon + 1, local_time.tm_mday,
            local_time.tm_hour, local_time.tm_min, local_time.tm_sec);

    fp = fopen(filename, "w");

    if(screen_yaml) {
      fprintf(stdout, "run_configuration: \n");
      fprintf(stdout, "  variant: " VARIANT_STRING "\n");
      fprintf(stdout, "  mpi_processes: %i\n", nprocs);
      fprintf(stdout, "  thread_teams: %i\n", nthreads);
      fprintf(stdout, "  threads: %i\n", nthreads);
      fprintf(stdout, "  datafile: %s\n", in.datafile ? in.datafile : "None");
      fprintf(stdout, "  units: %s\n", in.units == 0 ? "LJ" : "METAL");
      fprintf(stdout, "  atoms: %i\n", atom.natoms);
      fprintf(stdout, "  system_size: %2.2lf %2.2lf %2.2lf\n", atom.box.xprd, atom.box.yprd, atom.box.zprd);
      fprintf(stdout, "  unit_cells: %i %i %i\n", in.nx, in.ny, in.nz);
      fprintf(stdout, "  density: %lf\n", in.rho);
      fprintf(stdout, "  force_type: %s\n", in.forcetype == FORCELJ ? "LJ" : "EAM");
      fprintf(stdout, "  force_cutoff: %lf\n", force->cutforce);
      fprintf(stdout, "  force_params: %2.2lf %2.2lf\n",force->epsilon_scalar,force->sigma_scalar);
      fprintf(stdout, "  neighbor_cutoff: %lf\n", neighbor.cutneigh);
      fprintf(stdout, "  neighbor_type: %i\n", neighbor.halfneigh);
      fprintf(stdout, "  neighbor_team_build: %i\n", neighbor.team_neigh_build);
      fprintf(stdout, "  neighbor_bins: %i %i %i\n", neighbor.nbinx, neighbor.nbiny, neighbor.nbinz);
      fprintf(stdout, "  neighbor_frequency: %i\n", neighbor.every);
      fprintf(stdout, "  sort_frequency: %i\n", integrate.sort_every);
      fprintf(stdout, "  timestep_size: %lf\n", integrate.dt);
      fprintf(stdout, "  thermo_frequency: %i\n", thermo.nstat);
      fprintf(stdout, "  ghost_newton: %i\n", neighbor.ghost_newton);
      fprintf(stdout, "  use_intrinsics: %i\n", force->use_sse);
      fprintf(stdout, "  safe_exchange: %i\n", comm.do_safeexchange);
      fprintf(stdout, "  float_size: %i\n\n", (int) sizeof(MMD_float));
    }

    fprintf(fp, "run_configuration: \n");
    fprintf(fp, "  variant: " VARIANT_STRING "\n");
    fprintf(fp, "  mpi_processes: %i\n", nprocs);
    fprintf(fp, "  thread_teams: %i\n", nthreads);
    fprintf(fp, "  threads: %i\n", nthreads);
    fprintf(fp, "  datafile: %s\n", in.datafile ? in.datafile : "None");
    fprintf(fp, "  units: %s\n", in.units == 0 ? "LJ" : "METAL");
    fprintf(fp, "  atoms: %i\n", atom.natoms);
    fprintf(fp, "  system_size: %2.2lf %2.2lf %2.2lf\n", atom.box.xprd, atom.box.yprd, atom.box.zprd);
    fprintf(fp, "  unit_cells: %i %i %i\n", in.nx, in.ny, in.nz);
    fprintf(fp, "  density: %lf\n", in.rho);
    fprintf(fp, "  force_type: %s\n", in.forcetype == FORCELJ ? "LJ" : "EAM");
    fprintf(fp, "  force_cutoff: %lf\n", force->cutforce);
    fprintf(fp, "  force_params: %2.2lf %2.2lf\n",force->epsilon_scalar,force->sigma_scalar);
    fprintf(fp, "  neighbor_cutoff: %lf\n", neighbor.cutneigh);
    fprintf(fp, "  neighbor_type: %i\n", neighbor.halfneigh);
    fprintf(fp, "  neighbor_team_build: %i\n", neighbor.team_neigh_build);
    fprintf(fp, "  neighbor_bins: %i %i %i\n", neighbor.nbinx, neighbor.nbiny, neighbor.nbinz);
    fprintf(fp, "  neighbor_frequency: %i\n", neighbor.every);
    fprintf(fp, "  sort_frequency: %i\n", integrate.sort_every);
    fprintf(fp, "  timestep_size: %lf\n", integrate.dt);
    fprintf(fp, "  thermo_frequency: %i\n", thermo.nstat);
    fprintf(fp, "  ghost_newton: %i\n", neighbor.ghost_newton);
    fprintf(fp, "  use_intrinsics: %i\n", force->use_sse);
    fprintf(fp, "  safe_exchange: %i\n", comm.do_safeexchange);
    fprintf(fp, "  float_size: %i\n\n", (int) sizeof(MMD_float));

    if(screen_yaml)
      fprintf(stdout, "\n\nthermodynamic_output:\n");

    fprintf(fp, "\n\nthermodynamic_output:\n");

    for(i = 0; i < thermo.mstat; i++) {
      conserve = (1.5 * thermo.tmparr[i] + thermo.engarr[i]) /
                 (1.5 * thermo.tmparr[0] + thermo.engarr[0]);

      if(screen_yaml) {
        fprintf(stdout, "  timestep: %d \n", thermo.steparr[i]);
        fprintf(stdout, "      T*:           %15.10g \n", thermo.tmparr[i]);
        //fprintf(stdout,"      U*:           %15.10g \n", thermo.engarr[i]+engcorr);
        //fprintf(stdout,"      P*:           %15.10g \n", thermo.prsarr[i]+prscorr);
        fprintf(stdout, "      U*:           %15.10g \n", thermo.engarr[i]);
        fprintf(stdout, "      P*:           %15.10g \n", thermo.prsarr[i]);
        fprintf(stdout, "      Conservation: %15.10g \n", conserve);
      }

      fprintf(fp    , "  timestep: %d \n", thermo.steparr[i]);
      fprintf(fp    , "      T*:           %15.10g \n", thermo.tmparr[i]);
      //fprintf(fp    ,"      U*:           %15.10g \n", thermo.engarr[i]+engcorr);
      //fprintf(fp    ,"      P*:           %15.10g \n", thermo.prsarr[i]+prscorr);
      fprintf(fp    , "      U*:           %15.10g \n", thermo.engarr[i]);
      fprintf(fp    , "      P*:           %15.10g \n", thermo.prsarr[i]);
      fprintf(fp    , "      Conservation: %15.10g \n", conserve);
    }
  }

  /* performance output */

  if(me == 0) {
    fprintf(stdout, "\n\n");
    fprintf(fp, "\n\n");
  }

  double time_total = timer.array[TIME_TOTAL];
  MPI_Allreduce(&time_total, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  time_total = tmp / nprocs;
  double mflops = 4.0 / 3.0 * 3.1415926 *
                  pow(force->cutforce, MMD_float(3.0)) * in.rho * 0.5 *
                  23 * natoms * integrate.ntimes / time_total / 1000000.0;

  if(me == 0) {
    if(screen_yaml) {
      fprintf(stdout, "time:\n");
      fprintf(stdout, "  total:\n");
      fprintf(stdout, "    time: %g \n", time_total);
      fprintf(stdout, "    performance: %10.5e \n", natoms * integrate.ntimes / time_total);
      fprintf(stdout, "    performance_proc: %10.5e \n", natoms * integrate.ntimes / time_total / nprocs / nthreads);
    }

    fprintf(fp,    "time:\n");
    fprintf(fp,    "  total:\n");
    fprintf(fp,    "    time: %g \n", time_total);
    fprintf(fp,    "    performance: %10.5e \n", natoms * integrate.ntimes / time_total);
    fprintf(fp,    "    performance_proc: %10.5e \n", natoms * integrate.ntimes / time_total / nprocs / nthreads);
  }

  if(time_total == 0.0) time_total = 1.0;

  double time_force = timer.array[TIME_FORCE];
  MPI_Allreduce(&time_force, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tmp /= nprocs;

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "  force: %g\n", tmp);

    fprintf(fp,    "  force: %g\n", tmp);
  }

  double time_neigh = timer.array[TIME_NEIGH];
  MPI_Allreduce(&time_neigh, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tmp /= nprocs;

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "  neigh: %g\n", tmp);

    fprintf(fp,    "  neigh: %g\n", tmp);
  }

  double time_comm = timer.array[TIME_COMM];
  MPI_Allreduce(&time_comm, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tmp /= nprocs;

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "  comm:  %g\n", tmp);

    fprintf(fp,    "  comm:  %g\n", tmp);
  }


  double time_other = time_total - (time_force + time_neigh + time_comm);
  MPI_Allreduce(&time_other, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tmp /= nprocs;

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "  other: %g\n", tmp);

    fprintf(fp,    "  other: %g\n", tmp);
  }

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "\n");

    fprintf(fp, "\n");
  }

  stats(1, &time_force, &ave, &max, &min, 10, histo);

  if(me == 0) {
    if(screen_yaml) {
      fprintf(stdout, "# Timing histograms \n");
      fprintf(stdout, "# Force time: %g ave %g max %g min\n", ave, max, min);
      fprintf(stdout, "# Histogram:");

      for(i = 0; i < 10; i++) fprintf(stdout, " %d", histo[i]);

      fprintf(stdout, "\n");
    }

    fprintf(fp, "# Timing histograms \n");
    fprintf(fp, "# Force time: %g ave %g max %g min\n", ave, max, min);
    fprintf(fp, "# Histogram:");

    for(i = 0; i < 10; i++) fprintf(fp, " %d", histo[i]);

    fprintf(fp, "\n");
  }

  stats(1, &time_neigh, &ave, &max, &min, 10, histo);

  if(me == 0) {
    if(screen_yaml) {
      fprintf(stdout, "# Neigh time: %g ave %g max %g min\n", ave, max, min);
      fprintf(stdout, "# Histogram:");

      for(i = 0; i < 10; i++) fprintf(stdout, " %d", histo[i]);

      fprintf(stdout, "\n");
    }

    fprintf(fp, "# Neigh time: %g ave %g max %g min\n", ave, max, min);
    fprintf(fp, "# Histogram:");

    for(i = 0; i < 10; i++) fprintf(fp, " %d", histo[i]);

    fprintf(fp, "\n");
  }

  stats(1, &time_comm, &ave, &max, &min, 10, histo);

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "# Comm  time: %g ave %g max %g min\n", ave, max, min);

    if(screen_yaml)
      fprintf(stdout, "# Histogram:");

    if(screen_yaml)
      for(i = 0; i < 10; i++) fprintf(stdout, " %d", histo[i]);

    if(screen_yaml)
      fprintf(stdout, "\n");

    fprintf(fp, "# Comm  time: %g ave %g max %g min\n", ave, max, min);
    fprintf(fp, "# Histogram:");

    for(i = 0; i < 10; i++) fprintf(fp, " %d", histo[i]);

    fprintf(fp, "\n");
  }

  stats(1, &time_other, &ave, &max, &min, 10, histo);

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "# Other time: %g ave %g max %g min\n", ave, max, min);

    if(screen_yaml)
      fprintf(stdout, "# Histogram:");

    if(screen_yaml)
      for(i = 0; i < 10; i++) fprintf(stdout, " %d", histo[i]);

    if(screen_yaml)
      fprintf(stdout, "\n");

    fprintf(fp, "# Other time: %g ave %g max %g min\n", ave, max, min);
    fprintf(fp, "# Histogram:");

    for(i = 0; i < 10; i++) fprintf(fp, " %d", histo[i]);

    fprintf(fp, "\n");
  }

  if(me == 0) {
    fprintf(stdout, "\n");
    fprintf(fp, "\n");
  }

  tmp = atom.nlocal;
  stats(1, &tmp, &ave, &max, &min, 10, histo);

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "# Nlocal:     %g ave %g max %g min\n", ave, max, min);

    if(screen_yaml)
      fprintf(stdout, "# Histogram:");

    if(screen_yaml)
      for(i = 0; i < 10; i++) fprintf(stdout, " %d", histo[i]);

    if(screen_yaml)
      fprintf(stdout, "\n");

    fprintf(fp, "# Nlocal:     %g ave %g max %g min\n", ave, max, min);
    fprintf(fp, "# Histogram:");

    for(i = 0; i < 10; i++) fprintf(fp, " %d", histo[i]);

    fprintf(fp, "\n");
  }

  tmp = atom.nghost;
  stats(1, &tmp, &ave, &max, &min, 10, histo);

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "# Nghost:     %g ave %g max %g min\n", ave, max, min);

    if(screen_yaml)
      fprintf(stdout, "# Histogram:");

    if(screen_yaml)
      for(i = 0; i < 10; i++) fprintf(stdout, " %d", histo[i]);

    if(screen_yaml)
      fprintf(stdout, "\n");

    fprintf(fp, "# Nghost:     %g ave %g max %g min\n", ave, max, min);
    fprintf(fp, "# Histogram:");

    for(i = 0; i < 10; i++) fprintf(fp, " %d", histo[i]);

    fprintf(fp, "\n");
  }

  n = 0;

  for(i = 0; i < comm.nswap; i++) n += comm.sendnum[i];

  tmp = n;
  stats(1, &tmp, &ave, &max, &min, 10, histo);

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "# Nswaps:     %g ave %g max %g min\n", ave, max, min);

    if(screen_yaml)
      fprintf(stdout, "# Histogram:");

    if(screen_yaml)
      for(i = 0; i < 10; i++) fprintf(stdout, " %d", histo[i]);

    if(screen_yaml)
      fprintf(stdout, "\n");

    fprintf(fp, "# Nswaps:     %g ave %g max %g min\n", ave, max, min);
    fprintf(fp, "# Histogram:");

    for(i = 0; i < 10; i++) fprintf(fp, " %d", histo[i]);

    fprintf(fp, "\n");
  }

  n = 0;

  for(i = 0; i < atom.nlocal; i++) n += neighbor.numneigh[i];

  tmp = n;
  stats(1, &tmp, &ave, &max, &min, 10, histo);

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "# Neighs:     %g ave %g max %g min\n", ave, max, min);

    if(screen_yaml)
      fprintf(stdout, "# Histogram:");

    if(screen_yaml)
      for(i = 0; i < 10; i++) fprintf(stdout, " %d", histo[i]);

    if(screen_yaml)
      fprintf(stdout, "\n");

    fprintf(fp, "# Neighs:     %g ave %g max %g min\n", ave, max, min);
    fprintf(fp, "# Histogram:");

    for(i = 0; i < 10; i++) fprintf(fp, " %d", histo[i]);

    fprintf(fp, "\n");
  }

  MPI_Allreduce(&tmp, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "# Total # of neighbors = %g\n", total);

    fprintf(fp, "# Total # of neighbors = %g\n", total);
  }

  if(me == 0) {
    if(screen_yaml)
      fprintf(stdout, "\n");

    fprintf(fp, "\n");
  }

  if(me == 0) fclose(fp);
}

void stats(int n, double* data, double* pave, double* pmax, double* pmin,
           int nhisto, int* histo)
{
  int i, m;
  int* histotmp;

  double min = 1.0e20;
  double max = -1.0e20;
  double ave = 0.0;

  for(i = 0; i < n; i++) {
    ave += data[i];

    if(data[i] < min) min = data[i];

    if(data[i] > max) max = data[i];
  }

  int ntotal;
  MPI_Allreduce(&n, &ntotal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  double tmp;
  MPI_Allreduce(&ave, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  ave = tmp / ntotal;
  MPI_Allreduce(&min, &tmp, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  min = tmp;
  MPI_Allreduce(&max, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  max = tmp;

  for(i = 0; i < nhisto; i++) histo[i] = 0;

  double del = max - min;

  for(i = 0; i < n; i++) {
    if(del == 0.0) m = 0;
    else m = static_cast<int>((data[i] - min) / del * nhisto);

    if(m > nhisto - 1) m = nhisto - 1;

    histo[m]++;
  }

  histotmp = new int[nhisto];
  MPI_Allreduce(histo, histotmp, nhisto, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  for(i = 0; i < nhisto; i++) histo[i] = histotmp[i];

  delete [] histotmp;

  *pave = ave;
  *pmax = max;
  *pmin = min;
}
