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

#include <cstdio>
#include <cmath>
#include "mpi.h"
#include "atom.h"
#include "thermo.h"
#include "types.h"
#include "integrate.h"
#include "neighbor.h"

#include <cstring>
#include <cstdio>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

double random(int*);

#define NSECTIONS 3
#define MAXLINE 255
char line[MAXLINE];
char keyword[MAXLINE];
FILE* fp;

void read_lammps_parse_keyword(int first)
{
  int eof = 0;
  char buffer[MAXLINE];

  // proc 0 reads upto non-blank line plus 1 following line
  // eof is set to 1 if any read hits end-of-file

  if(!first) {
    if(fgets(line, MAXLINE, fp) == NULL) eof = 1;
  }

  while(eof == 0 && strspn(line, " \t\n\r") == strlen(line)) {
    if(fgets(line, MAXLINE, fp) == NULL) eof = 1;
  }

  if(fgets(buffer, MAXLINE, fp) == NULL) eof = 1;

  // if eof, set keyword empty and return

  if(eof) {
    keyword[0] = '\0';
    return;
  }

  // bcast keyword line to all procs


  // copy non-whitespace portion of line into keyword

  int start = strspn(line, " \t\n\r");
  int stop = strlen(line) - 1;

  while(line[stop] == ' ' || line[stop] == '\t'
        || line[stop] == '\n' || line[stop] == '\r') stop--;

  line[stop + 1] = '\0';
  strcpy(keyword, &line[start]);
}

void read_lammps_header(Atom &atom)
{
  int n;
  char* ptr;

  // customize for new sections

  const char* section_keywords[NSECTIONS] =
  {"Atoms", "Velocities", "Masses"};

  // skip 1st line of file

  char* eof = fgets(line, MAXLINE, fp);

  // customize for new header lines
  int ntypes = 0;

  while(1) {

    if(fgets(line, MAXLINE, fp) == NULL) n = 0;
    else n = strlen(line) + 1;

    if(n == 0) {
      line[0] = '\0';
      return;
    }

    // trim anything from '#' onward
    // if line is blank, continue

    double xlo, xhi, ylo, yhi, zlo, zhi;

    if(ptr = strchr(line, '#')) * ptr = '\0';

    if(strspn(line, " \t\n\r") == strlen(line)) continue;

    // search line for header keyword and set corresponding variable

    if(strstr(line, "atoms")) sscanf(line, "%i", &atom.natoms);
    else if(strstr(line, "atom types")) sscanf(line, "%i", &ntypes);

    // check for these first
    // otherwise "triangles" will be matched as "angles"

    else if(strstr(line, "xlo xhi")) {
      sscanf(line, "%lg %lg", &xlo, &xhi);
      atom.box.xprd = xhi - xlo;
    } else if(strstr(line, "ylo yhi")) {
      sscanf(line, "%lg %lg", &ylo, &yhi);
      atom.box.yprd = yhi - ylo;
    } else if(strstr(line, "zlo zhi")) {
      sscanf(line, "%lg %lg", &zlo, &zhi);
      atom.box.zprd = zhi - zlo;
    } else break;
  }

  // error check on total system size


  // check that exiting string is a valid section keyword

  read_lammps_parse_keyword(1);

  for(n = 0; n < NSECTIONS; n++)
    if(strcmp(keyword, section_keywords[n]) == 0) break;

  if(n == NSECTIONS) {
    char str[128];
    sprintf(str, "Unknown identifier in data file: %s", keyword);
  }

  // error check on consistency of header values
}

void read_lammps_atoms(Atom &atom, x_host_view_type x)
{
  int i;

  int nread = 0;
  int natoms = atom.natoms;
  atom.nlocal = 0;

  int type;
  double xx, xy, xz;

  while(nread < natoms) {
    fgets(line, MAXLINE, fp);
    sscanf(line, "%i %i %lg %lg %lg", &i, &type, &xx, &xy, &xz);
    i--;
    x(i,0) = xx;
    x(i,1) = xy;
    x(i,2) = xz;
    nread++;
  }

}

void read_lammps_velocities(Atom &atom, x_host_view_type v)
{
  int i;

  int nread = 0;
  int natoms = atom.natoms;

  double x, y, z;

  while(nread < natoms) {
    fgets(line, MAXLINE, fp);
    sscanf(line, "%i %lg %lg %lg", &i, &x, &y, &z);
    i--;
    v(i,0) = x;
    v(i,1) = y;
    v(i,2) = z;
    nread++;
  }

  // check that all atoms were assigned correctly

}

int read_lammps_data(Atom &atom, Comm &comm, Neighbor &neighbor, Integrate &integrate, Thermo &thermo, char* file, int units)
{
  fp = fopen(file, "r");

  if(fp == NULL) {
    char str[128];
    sprintf(str, "Cannot open file %s", file);
  }

  read_lammps_header(atom);

  comm.setup(neighbor.cutneigh, atom);

  if(neighbor.nbinx < 0) {
    MMD_float volume = atom.box.xprd * atom.box.yprd * atom.box.zprd;
    MMD_float rho = 1.0 * atom.natoms / volume;
    MMD_float neigh_bin_size = pow(rho * 16, MMD_float(1.0 / 3.0));
    neighbor.nbinx = atom.box.xprd / neigh_bin_size;
    neighbor.nbiny = atom.box.yprd / neigh_bin_size;
    neighbor.nbinz = atom.box.zprd / neigh_bin_size;
  }

  if(neighbor.nbinx == 0) neighbor.nbinx = 1;

  if(neighbor.nbiny == 0) neighbor.nbiny = 1;

  if(neighbor.nbinz == 0) neighbor.nbinz = 1;

  neighbor.setup(atom);

  integrate.setup();

  //force->setup();

  thermo.setup(atom.box.xprd * atom.box.yprd * atom.box.zprd / atom.natoms, integrate, atom, units);

  x_host_view_type x("Setup::x",atom.natoms);
  x_host_view_type v("Setup::v",atom.natoms);

  int atomflag = 0;
  int tmp;

  while(strlen(keyword)) {
    if(strcmp(keyword, "Atoms") == 0) {
      read_lammps_atoms(atom, x);
      atomflag = 1;
    } else if(strcmp(keyword, "Velocities") == 0) {
      if(atomflag == 0) printf("Must read Atoms before Velocities\n");

      read_lammps_velocities(atom, v);
    } else if(strcmp(keyword, "Masses") == 0) {
      fgets(line, MAXLINE, fp);

#if PRECISION==1
        sscanf(line, "%i %g", &tmp, &atom.mass);
#else
        sscanf(line, "%i %lg", &tmp, &atom.mass);
#endif
    }

    read_lammps_parse_keyword(0);
  }

  for(int i = 0; i < atom.natoms; i++) {
    if(x(i,0) >= atom.box.xlo && x(i,0) < atom.box.xhi &&
        x(i,1) >= atom.box.ylo && x(i,1) < atom.box.yhi &&
        x(i,2) >= atom.box.zlo && x(i,2) < atom.box.zhi)
      atom.addatom(x(i,0), x(i,1), x(i,2), v(i,0), v(i,1), v(i,2));
  }

  int me;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  /* check that correct # of atoms were created */

  int natoms;
  MPI_Allreduce(&atom.nlocal, &natoms, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(natoms != atom.natoms) {
    if(me == 0) printf("Created incorrect # of atoms\n");

    return 1;
  }

  Kokkos::deep_copy(atom.x,atom.h_x);
  Kokkos::deep_copy(atom.v,atom.h_v);
  Kokkos::deep_copy(atom.type,atom.h_type);
  // check that all atoms were assigned correctly
  return 0;
}

/* create simulation box */

void create_box(Atom &atom, int nx, int ny, int nz, double rho)
{
  double lattice = pow((4.0 / rho), (1.0 / 3.0));
  atom.box.xprd = nx * lattice;
  atom.box.yprd = ny * lattice;
  atom.box.zprd = nz * lattice;
}

/* initialize atoms on fcc lattice in parallel fashion */

int create_atoms(Atom &atom, int nx, int ny, int nz, double rho)
{
  /* total # of atoms */

  atom.natoms = 4 * nx * ny * nz;
  atom.nlocal = 0;

  /* determine loop bounds of lattice subsection that overlaps my sub-box
     insure loop bounds do not exceed nx,ny,nz */

  double alat = pow((4.0 / rho), (1.0 / 3.0));
  int ilo = static_cast<int>(atom.box.xlo / (0.5 * alat) - 1);
  int ihi = static_cast<int>(atom.box.xhi / (0.5 * alat) + 1);
  int jlo = static_cast<int>(atom.box.ylo / (0.5 * alat) - 1);
  int jhi = static_cast<int>(atom.box.yhi / (0.5 * alat) + 1);
  int klo = static_cast<int>(atom.box.zlo / (0.5 * alat) - 1);
  int khi = static_cast<int>(atom.box.zhi / (0.5 * alat) + 1);

  ilo = MAX(ilo, 0);
  ihi = MIN(ihi, 2 * nx - 1);
  jlo = MAX(jlo, 0);
  jhi = MIN(jhi, 2 * ny - 1);
  klo = MAX(klo, 0);
  khi = MIN(khi, 2 * nz - 1);

  /* each proc generates positions and velocities of atoms on fcc sublattice
       that overlaps its box
     only store atoms that fall in my box
     use atom # (generated from lattice coords) as unique seed to generate a
       unique velocity
     exercise RNG between calls to avoid correlations in adjacent atoms */

  double xtmp, ytmp, ztmp, vx, vy, vz;
  int i, j, k, m, n;
  int sx = 0;
  int sy = 0;
  int sz = 0;
  int ox = 0;
  int oy = 0;
  int oz = 0;
  int subboxdim = 8;

  int iflag = 0;

  while(oz * subboxdim <= khi) {
    k = oz * subboxdim + sz;
    j = oy * subboxdim + sy;
    i = ox * subboxdim + sx;

    if(iflag) continue;

    if(((i + j + k) % 2 == 0) &&
        (i >= ilo) && (i <= ihi) &&
        (j >= jlo) && (j <= jhi) &&
        (k >= klo) && (k <= khi)) {

      xtmp = 0.5 * alat * i;
      ytmp = 0.5 * alat * j;
      ztmp = 0.5 * alat * k;

      if(xtmp >= atom.box.xlo && xtmp < atom.box.xhi &&
          ytmp >= atom.box.ylo && ytmp < atom.box.yhi &&
          ztmp >= atom.box.zlo && ztmp < atom.box.zhi) {
        n = k * (2 * ny) * (2 * nx) + j * (2 * nx) + i + 1;

        for(m = 0; m < 5; m++) random(&n);

        vx = random(&n);

        for(m = 0; m < 5; m++) random(&n);

        vy = random(&n);

        for(m = 0; m < 5; m++) random(&n);

        vz = random(&n);

        atom.addatom(xtmp, ytmp, ztmp, vx, vy, vz);
      }
    }

    sx++;

    if(sx == subboxdim) {
      sx = 0;
      sy++;
    }

    if(sy == subboxdim) {
      sy = 0;
      sz++;
    }

    if(sz == subboxdim) {
      sz = 0;
      ox++;
    }

    if(ox * subboxdim > ihi) {
      ox = 0;
      oy++;
    }

    if(oy * subboxdim > jhi) {
      oy = 0;
      oz++;
    }
  }

  /* check for overflows on any proc */

  int me;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  int iflagall;
  MPI_Allreduce(&iflag, &iflagall, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if(iflagall) {
    if(me == 0) printf("No memory for atoms\n");

    return 1;
  }

  /* check that correct # of atoms were created */

  int natoms;
  MPI_Allreduce(&atom.nlocal, &natoms, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(natoms != atom.natoms) {
    if(me == 0) printf("Created incorrect # of atoms\n");

    return 1;
  }
  Kokkos::deep_copy(atom.x,atom.h_x);
  Kokkos::deep_copy(atom.v,atom.h_v);
  Kokkos::deep_copy(atom.type,atom.h_type);

  return 0;
}

/* adjust initial velocities to give desired temperature */

void create_velocity(double t_request, Atom &atom, Thermo &thermo)
{
  int i;

  /* zero center-of-mass motion */
  Kokkos::deep_copy(atom.h_v,atom.v);
  double vxtot = 0.0;
  double vytot = 0.0;
  double vztot = 0.0;

  for(i = 0; i < atom.nlocal; i++) {
    vxtot += atom.h_v(i,0);
    vytot += atom.h_v(i,1);
    vztot += atom.h_v(i,2);
  }

  double tmp;
  MPI_Allreduce(&vxtot, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  vxtot = tmp / atom.natoms;
  MPI_Allreduce(&vytot, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  vytot = tmp / atom.natoms;
  MPI_Allreduce(&vztot, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  vztot = tmp / atom.natoms;

  for(i = 0; i < atom.nlocal; i++) {
    atom.h_v(i,0) -= vxtot;
    atom.h_v(i,1) -= vytot;
    atom.h_v(i,2) -= vztot;
  }

  /* rescale velocities, including old ones */
  thermo.t_act = 0;

  Kokkos::deep_copy(atom.v,atom.h_v);
  double t = thermo.temperature(atom);
  double factor = sqrt(t_request / t);
  Kokkos::deep_copy(atom.h_v,atom.v);

  for(i = 0; i < atom.nlocal; i++) {
    atom.h_v(i,0) *= factor;
    atom.h_v(i,1) *= factor;
    atom.h_v(i,2) *= factor;
  }
  Kokkos::deep_copy(atom.v,atom.h_v);
}

/* Park/Miller RNG w/out MASKING, so as to be like f90s version */

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

double random(int* idum)
{
  int k;
  double ans;

  k = (*idum) / IQ;
  *idum = IA * (*idum - k * IQ) - IR * k;

  if(*idum < 0) *idum += IM;

  ans = AM * (*idum);
  return ans;
}

#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef MASK
