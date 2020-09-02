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

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "force_eam.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "memory.h"

#define MAXLINE 1024

#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)

/* ---------------------------------------------------------------------- */

ForceEAM::ForceEAM(int ntypes_)
{
  ntypes = ntypes_;
  cutforce = 0.0;

  float_1d_view_type d_cut("ForceEAM::cutforcesq",ntypes*ntypes);
  float_1d_host_view_type h_cut = Kokkos::create_mirror_view(d_cut);
  cutforcesq = d_cut;

  for( int i = 0; i<ntypes*ntypes; i++)
    h_cut[i] = 0.0;

  Kokkos::deep_copy(d_cut,h_cut);

  use_oldcompute = 0;

  nmax = 0;

  style = FORCEEAM;

  nthreads = Kokkos::HostSpace::execution_space().concurrency();
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

ForceEAM::~ForceEAM()
{

}

void ForceEAM::setup()
{
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  coeff("Cu_u6.eam");
  init_style();
}


void ForceEAM::compute(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{
  if(neighbor.halfneigh)
    return compute_halfneigh(atom, neighbor, comm, me);
  else
    return compute_fullneigh(atom, neighbor, comm, me);

}
/* ---------------------------------------------------------------------- */

void ForceEAM::compute_halfneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{
  virial = 0;
  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  if(atom.nmax > nmax) {
    nmax = atom.nmax;
    rho = float_1d_view_type("PairEAM::rho",nmax);
    fp = float_1d_view_type("PairEAM::fp",nmax);
  }

  nlocal = atom.nlocal;
  nall = atom.nlocal + atom.nghost;

  x = atom.x;
  f = atom.f;
  type = atom.type;

  rho_a = rho;

  neighbors = neighbor.neighbors;
  numneigh = neighbor.numneigh;

  // zero out density

  Kokkos::deep_copy(f,0);
  Kokkos::deep_copy(rho,0);

  // rho = density at each atom
  // loop over neighbors of my atoms
  eng_virial_type t_eng_virial;


  Kokkos::parallel_for(Kokkos::RangePolicy<TagHalfNeighInitial >(0,nlocal), *this );

  // fp = derivative of embedding energy at each atom
  // phi = embedding energy at each atom

  if(evflag)
    Kokkos::parallel_reduce(Kokkos::RangePolicy<TagHalfNeighMiddle<1> >(0,nlocal), *this ,t_eng_virial);
  else
    Kokkos::parallel_for(Kokkos::RangePolicy<TagHalfNeighMiddle<0> >(0,nlocal), *this);

  // communicate derivative of embedding function

  communicate(atom, comm);


  // compute forces on each atom
  // loop over neighbors of my atoms
  eng_virial_type t_eng_virial_2;
  if(evflag)
    Kokkos::parallel_reduce(Kokkos::RangePolicy<TagHalfNeighFinal<1> >(0,nlocal), *this ,t_eng_virial_2);
  else
    Kokkos::parallel_for(Kokkos::RangePolicy<TagHalfNeighFinal<0> >(0,nlocal), *this);

  t_eng_virial += t_eng_virial_2;
  eng_vdwl = t_eng_virial.eng;
  virial = t_eng_virial.virial;
}

KOKKOS_INLINE_FUNCTION
void ForceEAM::operator() (TagHalfNeighInitial , const int& i ) const {
  const float_1d_atomic_um_view_type rho_ = rho;
  const int jnum = numneigh[i];
  const MMD_float xtmp = x(i,0);
  const MMD_float ytmp = x(i,1);
  const MMD_float ztmp = x(i,2);
  const int type_i = type[i];
  MMD_float rhoi = 0.0;

  for(MMD_int jj = 0; jj < jnum; jj++) {
    const MMD_int j = neighbors(i,jj);

    const MMD_float delx = xtmp - x(j,0);
    const MMD_float dely = ytmp - x(j,1);
    const MMD_float delz = ztmp - x(j,2);
    const int type_j = type[j];
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;

    const int type_ij = type_i*ntypes+type_j;

    if(rsq < cutforcesq(type_ij)) {
      MMD_float p = sqrt(rsq) * rdr + 1.0;
      MMD_int m = static_cast<int>(p);
      m = m < nr - 1 ? m : nr - 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;

      const MMD_float d_rho = ((rhor_spline(type_ij , m , 3)  * p +
                                rhor_spline(type_ij , m , 4)) * p +
                                rhor_spline(type_ij , m , 5)) * p +
                                rhor_spline(type_ij , m , 6);
      rhoi += d_rho;

      if(j < nlocal) {
        #ifdef KOKKOS_HAVE_SERIAL
        if(Kokkos::Impl::is_same<Kokkos::DefaultExecutionSpace,Kokkos::Serial>::value)
          rho(j) += d_rho;
        else
        #endif
          rho_a(j) += d_rho;
      }
    }
  }

  #ifdef KOKKOS_HAVE_SERIAL
  if(Kokkos::Impl::is_same<Kokkos::DefaultExecutionSpace,Kokkos::Serial>::value)
    rho(i) += rhoi;
  else
  #endif
    rho_a(i) += rhoi;
}


KOKKOS_INLINE_FUNCTION
void ForceEAM::operator() (TagHalfNeighMiddle<0> , const int& i ) const {
  eng_virial_type eng_virial;
  this->operator() (TagHalfNeighMiddle<0>(),i,eng_virial);
}

template<int EVFLAG>
KOKKOS_INLINE_FUNCTION
void ForceEAM::operator() (TagHalfNeighMiddle<EVFLAG> , const int& i, eng_virial_type& eng_virial ) const {
  MMD_float p = 1.0 * rho[i] * rdrho + 1.0;
  MMD_int m = static_cast<int>(p);
  const int type_ii = type[i] * type[i];
  m = MAX(1, MIN(m, nrho - 1));
  p -= m;
  p = MIN(p, 1.0);
  fp[i] = (frho_spline(type_ii , m , 0) * p +
           frho_spline(type_ii , m , 1)) * p +
           frho_spline(type_ii , m , 2);

  if(EVFLAG) {
    eng_virial.eng += ((frho_spline(type_ii , m , 3) * p +
                        frho_spline(type_ii , m , 4)) * p +
                        frho_spline(type_ii , m , 5)) * p +
                        frho_spline(type_ii , m , 6);
  }
}

KOKKOS_INLINE_FUNCTION
void ForceEAM::operator() (TagHalfNeighFinal<0> , const int& i ) const {
  eng_virial_type eng_virial;
  operator() (TagHalfNeighFinal<0>(),i,eng_virial);
}

template<int EVFLAG>
KOKKOS_INLINE_FUNCTION
void ForceEAM::operator() (TagHalfNeighFinal<EVFLAG> , const int& i, eng_virial_type& eng_virial ) const {
  x_atomic_um_view_type f_ = f;
  const int jnum = numneigh[i];
  const MMD_float xtmp = x(i,0);
  const MMD_float ytmp = x(i,1);
  const MMD_float ztmp = x(i,2);
  const int type_i = type[i];
  MMD_float fx = 0;
  MMD_float fy = 0;
  MMD_float fz = 0;

  for(MMD_int jj = 0; jj < jnum; jj++) {
    const MMD_int j = neighbors(i,jj);

    const MMD_float delx = xtmp - x(j,0);
    const MMD_float dely = ytmp - x(j,1);
    const MMD_float delz = ztmp - x(j,2);
    const int type_j = type[j];
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;

    const int type_ij = type_i*ntypes+type_j;

    if(rsq < cutforcesq(type_ij)) {
      MMD_float r = sqrt(rsq);
      MMD_float p = r * rdr + 1.0;
      MMD_int m = static_cast<int>(p);
      m = m < nr - 1 ? m : nr - 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;


      // rhoip = derivative of (density at atom j due to atom i)
      // rhojp = derivative of (density at atom i due to atom j)
      // phi = pair potential energy
      // phip = phi'
      // z2 = phi * r
      // z2p = (phi * r)' = (phi' r) + phi
      // psip needs both fp[i] and fp[j] terms since r_ij appears in two
      //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
      //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

      MMD_float rhoip = (rhor_spline(type_ij , m , 0) * p +
                         rhor_spline(type_ij , m , 1)) * p +
                         rhor_spline(type_ij , m , 2);
      MMD_float z2p = (z2r_spline(type_ij , m , 0) * p +
                       z2r_spline(type_ij , m , 1)) * p +
                       z2r_spline(type_ij , m , 2);
      MMD_float z2 = ((z2r_spline(type_ij , m , 3) * p +
                       z2r_spline(type_ij , m , 4)) * p +
                       z2r_spline(type_ij , m , 5)) * p +
                       z2r_spline(type_ij , m , 6);

      MMD_float recip = 1.0 / r;
      MMD_float phi = z2 * recip;
      MMD_float phip = z2p * recip - phi * recip;
      MMD_float psip = fp[i] * rhoip + fp[j] * rhoip + phip;
      MMD_float fpair = -psip * recip;

      fx += delx * fpair;
      fy += dely * fpair;
      fz += delz * fpair;

      if(j < nlocal) {
        #ifdef KOKKOS_HAVE_SERIAL
        if(Kokkos::Impl::is_same<Kokkos::DefaultExecutionSpace,Kokkos::Serial>::value) {
          f(j,0) -= delx * fpair;
          f(j,1) -= dely * fpair;
          f(j,2) -= delz * fpair;
        } else
        #endif
        {
          f_(j,0) -= delx * fpair;
          f_(j,1) -= dely * fpair;
          f_(j,2) -= delz * fpair;
        }
      } else fpair *= 0.5;

      if(EVFLAG) {
        eng_virial.virial += delx * delx * fpair + dely * dely * fpair + delz * delz * fpair;
      }

      if(j < nlocal) eng_virial.eng += phi;
      else eng_virial.eng += 0.5 * phi;
    }
  }

  #ifdef KOKKOS_HAVE_SERIAL
  if(Kokkos::Impl::is_same<Kokkos::DefaultExecutionSpace,Kokkos::Serial>::value) {
    f(i,0) += fx;
    f(i,1) += fy;
    f(i,2) += fz;
  } else
  #endif
  {
    f_(i,0) += fx;
    f_(i,1) += fy;
    f_(i,2) += fz;
  }
}
/* ---------------------------------------------------------------------- */

void ForceEAM::compute_fullneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{

  eng_virial_type t_eng_virial;

  virial = 0;
  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  if(atom.nmax > nmax) {
    nmax = atom.nmax;
    rho = float_1d_view_type("PairEAM::rho",nmax);
    rho_a = rho;
    fp = float_1d_view_type("PairEAM::fp",nmax);
  }

  nlocal = atom.nlocal;
  nall = atom.nlocal + atom.nghost;

  x = atom.x;
  f = atom.f;
  type = atom.type;

  neighbors = neighbor.neighbors;
  numneigh = neighbor.numneigh;

  // zero out density

  // rho = density at each atom
  // loop over neighbors of my atoms

  if(evflag)
    Kokkos::parallel_reduce(Kokkos::RangePolicy<TagFullNeighInitial<1> >(0,nlocal), *this ,t_eng_virial);
  else
    Kokkos::parallel_for(Kokkos::RangePolicy<TagFullNeighInitial<0> >(0,nlocal), *this );

  // fp = derivative of embedding energy at each atom
  // phi = embedding energy at each atom

  // communicate derivative of embedding function

  Kokkos::fence();
  communicate(atom, comm);


  eng_virial_type t_eng_virial_2;
  // compute forces on each atom
  // loop over neighbors of my atoms

  if(evflag)
    Kokkos::parallel_reduce(Kokkos::RangePolicy<TagFullNeighFinal<1> >(0,nlocal), *this ,t_eng_virial_2);
  else
    Kokkos::parallel_for(Kokkos::RangePolicy<TagFullNeighFinal<0> >(0,nlocal), *this );

  t_eng_virial += t_eng_virial_2;
  virial += t_eng_virial.virial;
  eng_vdwl += 2.0 * t_eng_virial.eng;
}

KOKKOS_INLINE_FUNCTION
void ForceEAM::operator() (TagFullNeighInitial<0> , const int& i ) const {
  eng_virial_type eng_virial;
  operator() (TagFullNeighInitial<0>(),i,eng_virial);
}

template<int EVFLAG>
KOKKOS_INLINE_FUNCTION
void ForceEAM::operator() (TagFullNeighInitial<EVFLAG> , const int& i, eng_virial_type& eng_virial ) const {
  const int jnum = numneigh[i];
  const MMD_float xtmp = x(i,0);
  const MMD_float ytmp = x(i,1);
  const MMD_float ztmp = x(i,2);
  const int type_i = type[i];
  MMD_float rhoi = 0;

  #pragma ivdep
  for(MMD_int jj = 0; jj < jnum; jj++) {
    const MMD_int j = neighbors(i,jj);

    const MMD_float delx = xtmp - x(j,0);
    const MMD_float dely = ytmp - x(j,1);
    const MMD_float delz = ztmp - x(j,2);
    const int type_j = type[j];
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;

    const int type_ij = type_i*ntypes+type_j;

    if(rsq < cutforcesq(type_ij)) {
      MMD_float p = sqrt(rsq) * rdr + 1.0;
      MMD_int m = static_cast<int>(p);
      m = m < nr - 1 ? m : nr - 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;

      rhoi += ((rhor_spline(type_ij , m , 3) * p +
                rhor_spline(type_ij , m , 4)) * p +
                rhor_spline(type_ij , m , 5)) * p +
                rhor_spline(type_ij , m , 6);
    }
  }

  const int type_ii = type_i*type_i;
  MMD_float p = 1.0 * rhoi * rdrho + 1.0;
  MMD_int m = static_cast<int>(p);
  m = MAX(1, MIN(m, nrho - 1));
  p -= m;
  p = MIN(p, 1.0);
  fp[i] = (frho_spline(type_ii , m , 0) * p +
           frho_spline(type_ii , m , 1)) * p +
           frho_spline(type_ii , m , 2);

  if(evflag) {
    eng_virial.eng += ((frho_spline(type_ii , m , 3) * p +
                        frho_spline(type_ii , m , 4)) * p +
                        frho_spline(type_ii , m , 5)) * p +
                        frho_spline(type_ii , m , 6);
  }
}

KOKKOS_INLINE_FUNCTION
void ForceEAM::operator() (TagFullNeighFinal<0> , const int& i ) const {
  eng_virial_type eng_virial;
  operator() (TagFullNeighFinal<0>(),i,eng_virial);
}

template<int EVFLAG>
KOKKOS_INLINE_FUNCTION
void ForceEAM::operator() (TagFullNeighFinal<EVFLAG> , const int& i, eng_virial_type& eng_virial ) const {
  const int jnum = numneigh[i];
  const MMD_float xtmp = x(i,0);
  const MMD_float ytmp = x(i,1);
  const MMD_float ztmp = x(i,2);
  const int type_i = type[i];

  MMD_float fx = 0.0;
  MMD_float fy = 0.0;
  MMD_float fz = 0.0;

  #pragma ivdep
  for(MMD_int jj = 0; jj < jnum; jj++) {
    const MMD_int j = neighbors(i,jj);

    const MMD_float delx = xtmp - x(j,0);
    const MMD_float dely = ytmp - x(j,1);
    const MMD_float delz = ztmp - x(j,2);
    const int type_j = type[j];
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;

    const int type_ij = type_i*ntypes+type_j;

    if(rsq < cutforcesq(type_ij)) {
      MMD_float r = sqrt(rsq);
      MMD_float p = r * rdr + 1.0;
      MMD_int m = static_cast<int>(p);
      m = m < nr - 1 ? m : nr - 1;
      p -= m;
      p = p < 1.0 ? p : 1.0;


      // rhoip = derivative of (density at atom j due to atom i)
      // rhojp = derivative of (density at atom i due to atom j)
      // phi = pair potential energy
      // phip = phi'
      // z2 = phi * r
      // z2p = (phi * r)' = (phi' r) + phi
      // psip needs both fp[i] and fp[j] terms since r_ij appears in two
      //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
      //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

      MMD_float rhoip = (rhor_spline(type_ij , m , 0) * p +
                         rhor_spline(type_ij , m , 1)) * p +
                         rhor_spline(type_ij , m , 2);
      MMD_float z2p = (z2r_spline(type_ij , m , 0) * p +
                       z2r_spline(type_ij , m , 1)) * p +
                       z2r_spline(type_ij , m , 2);
      MMD_float z2 = ((z2r_spline(type_ij , m , 3) * p +
                       z2r_spline(type_ij , m , 4)) * p +
                       z2r_spline(type_ij , m , 5)) * p +
                       z2r_spline(type_ij , m , 6);

      MMD_float recip = 1.0 / r;
      MMD_float phi = z2 * recip;
      MMD_float phip = z2p * recip - phi * recip;
      MMD_float psip = fp[i] * rhoip + fp[j] * rhoip + phip;
      MMD_float fpair = -psip * recip;

      fx += delx * fpair;
      fy += dely * fpair;
      fz += delz * fpair;

      fpair *= 0.5;

      if(evflag) {
        eng_virial.virial += delx * delx * fpair + dely * dely * fpair + delz * delz * fpair;
        eng_virial.eng += 0.5 * phi;
      }

    }
  }

  f(i,0) = fx;
  f(i,1) = fy;
  f(i,2) = fz;
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   read DYNAMO funcfl file
------------------------------------------------------------------------- */

void ForceEAM::coeff(const char* arg)
{



  // read funcfl file if hasn't already been read
  // store filename in Funcfl data struct


  read_file(arg);
  int n = strlen(arg) + 1;
  funcfl.file = new char[n];

  // set setflag and map only for i,i type pairs
  // set mass of atom type if i = j

  //atom->mass = funcfl.mass;
  cutmax = funcfl.cut;

  float_1d_view_type d_cut("ForceEAM::cutforcesq",ntypes*ntypes);
  float_1d_host_view_type h_cut = Kokkos::create_mirror_view(d_cut);
  cutforcesq = d_cut;

  for( int i = 0; i<ntypes*ntypes; i++)
    h_cut[i] = cutmax * cutmax;

  Kokkos::deep_copy(d_cut,h_cut);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void ForceEAM::init_style()
{
  // convert read-in file(s) to arrays and spline them

  file2array();
  array2spline();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */



/* ----------------------------------------------------------------------
   read potential values from a DYNAMO single element funcfl file
------------------------------------------------------------------------- */

void ForceEAM::read_file(const char* filename)
{
  Funcfl* file = &funcfl;

  //me = 0;
  FILE* fptr;
  char line[MAXLINE];

  int flag = 0;

  if(me == 0) {
    fptr = fopen(filename, "r");

    if(fptr == NULL) {
      printf("Can't open EAM Potential file: %s\n", filename);
      flag = 1;
    }
  }

  MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(flag) {
    MPI_Finalize();
    exit(0);
  }

  int tmp;

  if(me == 0) {
    fgets(line, MAXLINE, fptr);
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%d %lg", &tmp, &file->mass);
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%d %lg %d %lg %lg",
           &file->nrho, &file->drho, &file->nr, &file->dr, &file->cut);
  }

  MPI_Bcast(&file->mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->nrho, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->drho, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->nr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->dr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->cut, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  mass = file->mass;
  file->frho = new MMD_float[file->nrho + 1];
  file->rhor = new MMD_float[file->nr + 1];
  file->zr = new MMD_float[file->nr + 1];

  if(me == 0) grab(fptr, file->nrho, file->frho);

  if(sizeof(MMD_float) == 4)
    MPI_Bcast(file->frho, file->nrho, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(file->frho, file->nrho, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(me == 0) grab(fptr, file->nr, file->zr);

  if(sizeof(MMD_float) == 4)
    MPI_Bcast(file->zr, file->nr, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(file->zr, file->nr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(me == 0) grab(fptr, file->nr, file->rhor);

  if(sizeof(MMD_float) == 4)
    MPI_Bcast(file->rhor, file->nr, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(file->rhor, file->nr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for(int i = file->nrho; i > 0; i--) file->frho[i] = file->frho[i - 1];

  for(int i = file->nr; i > 0; i--) file->rhor[i] = file->rhor[i - 1];

  for(int i = file->nr; i > 0; i--) file->zr[i] = file->zr[i - 1];

  if(me == 0) fclose(fptr);
}

/* ----------------------------------------------------------------------
   convert read-in funcfl potential(s) to standard array format
   interpolate all file values to a single grid and cutoff
------------------------------------------------------------------------- */

void ForceEAM::file2array()
{
  int k;
  double sixth = 1.0 / 6.0;

  // determine max function params from all active funcfl files
  // active means some element is pointing at it via map

  double rmax, rhomax;
  dr = drho = rmax = rhomax = 0.0;

  Funcfl* file = &funcfl;
  dr = MAX(dr, file->dr);
  drho = MAX(drho, file->drho);
  rmax = MAX(rmax, (file->nr - 1) * file->dr);
  rhomax = MAX(rhomax, (file->nrho - 1) * file->drho);

  // set nr,nrho from cutoff and spacings
  // 0.5 is for round-off in divide

  nr = static_cast<int>(rmax / dr + 0.5);
  nrho = static_cast<int>(rhomax / drho + 0.5);

  // ------------------------------------------------------------------
  // setup frho arrays
  // ------------------------------------------------------------------

  // allocate frho arrays
  // nfrho = # of funcfl files + 1 for zero array

  frho = new MMD_float[nrho + 1];

  // interpolate each file's frho to a single grid and cutoff

  double r, p, cof1, cof2, cof3, cof4;

  for(int m = 1; m <= nrho; m++) {
    r = (m - 1) * drho;
    p = r / file->drho + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, file->nrho - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    frho[m] = cof1 * file->frho[k - 1] + cof2 * file->frho[k] +
              cof3 * file->frho[k + 1] + cof4 * file->frho[k + 2];
  }


  // ------------------------------------------------------------------
  // setup rhor arrays
  // ------------------------------------------------------------------

  // allocate rhor arrays
  // nrhor = # of funcfl files

  rhor = new MMD_float[nr + 1];

  // interpolate each file's rhor to a single grid and cutoff

  for(int m = 1; m <= nr; m++) {
    r = (m - 1) * dr;
    p = r / file->dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, file->nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    rhor[m] = cof1 * file->rhor[k - 1] + cof2 * file->rhor[k] +
              cof3 * file->rhor[k + 1] + cof4 * file->rhor[k + 2];
  }

  // type2rhor[i][j] = which rhor array (0 to nrhor-1) each type pair maps to
  // for funcfl files, I,J mapping only depends on I
  // OK if map = -1 (non-EAM atom in pair hybrid) b/c type2rhor not used

  // ------------------------------------------------------------------
  // setup z2r arrays
  // ------------------------------------------------------------------

  // allocate z2r arrays
  // nz2r = N*(N+1)/2 where N = # of funcfl files

  z2r = new MMD_float[nr + 1];

  // create a z2r array for each file against other files, only for I >= J
  // interpolate zri and zrj to a single grid and cutoff

  double zri, zrj;

  Funcfl* ifile = &funcfl;
  Funcfl* jfile = &funcfl;

  for(int m = 1; m <= nr; m++) {
    r = (m - 1) * dr;

    p = r / ifile->dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, ifile->nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    zri = cof1 * ifile->zr[k - 1] + cof2 * ifile->zr[k] +
          cof3 * ifile->zr[k + 1] + cof4 * ifile->zr[k + 2];

    p = r / jfile->dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, jfile->nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    zrj = cof1 * jfile->zr[k - 1] + cof2 * jfile->zr[k] +
          cof3 * jfile->zr[k + 1] + cof4 * jfile->zr[k + 2];

    z2r[m] = 27.2 * 0.529 * zri * zrj;
  }

}

/* ---------------------------------------------------------------------- */

void ForceEAM::array2spline()
{
  rdr = 1.0 / dr;
  rdrho = 1.0 / drho;

  spline_dv_type dv_frho_spline("ForceEam::froh_spline",ntypes * ntypes , nrho+1);
  spline_dv_type dv_rhor_spline("ForceEam::rohr_spline",ntypes * ntypes , nr+1);
  spline_dv_type dv_z2r_spline("ForceEam::z2r_spline",ntypes * ntypes , nr+1);

  interpolate(nrho, drho, frho, dv_frho_spline.h_view);

  interpolate(nr, dr, rhor, dv_rhor_spline.h_view);

  interpolate(nr, dr, z2r, dv_z2r_spline.h_view);

  // replicate data for multiple types;
  for(int tt = 1 ; tt<ntypes*ntypes; tt++) {
    for(int k = 0; k<nrho+1; k++)
      for(int l=0;l<7;l++) {
        dv_frho_spline.h_view(tt,k,l) = dv_frho_spline.h_view(0,k,l);
      }
    for(int k = 0; k<nr+1; k++)
      for(int l=0;l<7;l++)
        dv_rhor_spline.h_view(tt,k,l) = dv_rhor_spline.h_view(0,k,l);
    for(int k = 0; k<nr+1; k++)
      for(int l=0;l<7;l++)
        dv_z2r_spline.h_view(tt,k,l) = dv_z2r_spline.h_view(0,k,l);
  }

  dv_frho_spline.modify<HostType>();
  dv_frho_spline.sync<DeviceType>();
  frho_spline = dv_frho_spline.d_view;

  dv_rhor_spline.modify<HostType>();
  dv_rhor_spline.sync<DeviceType>();
  rhor_spline = dv_rhor_spline.d_view;

  dv_z2r_spline.modify<HostType>();
  dv_z2r_spline.sync<DeviceType>();
  z2r_spline = dv_z2r_spline.d_view;

}

/* ---------------------------------------------------------------------- */

void ForceEAM::interpolate(MMD_int n, MMD_float delta, MMD_float* f, spline_host_type spline)
{
  for(int m = 1; m <= n; m++) spline(0 , m , 6) = f[m];

  spline(0 , 1 , 5) = spline(0 , 2 , 6) - spline(0 , 1 , 6);
  spline(0 , 2 , 5) = 0.5 * (spline(0 , 3 , 6) - spline(0 , 1 , 6));
  spline(0 , (n - 1) , 5) = 0.5 * (spline(0 , n , 6) - spline(0 , (n - 2) , 6));
  spline(0 , n , 5) = spline(0 , n , 6) - spline(0 , (n - 1) , 6);

  for(int m = 3; m <= n - 2; m++)
    spline(0 , m , 5) = ((spline(0 , (m - 2) , 6) - spline(0 , (m + 2) , 6)) +
                         8.0 * (spline(0 , (m + 1) , 6) - spline(0 , (m - 1) , 6))) / 12.0;

  for(int m = 1; m <= n - 1; m++) {
    spline(0 , m , 4) = 3.0 * (spline(0 , (m + 1) , 6) - spline(0 , m , 6)) -
                        2.0 * spline(0 , m , 5) - spline(0 , (m + 1) , 5);
    spline(0 , m , 3) = spline(0 , m , 5) + spline(0 , (m + 1) , 5) -
                        2.0 * (spline(0 , (m + 1) , 6) - spline(0 , m , 6));
  }

  spline(0 , n , 4) = 0.0;
  spline(0 , n , 3) = 0.0;

  for(int m = 1; m <= n; m++) {
    spline(0 , m , 2) = spline(0 , m , 5) / delta;
    spline(0 , m , 1) = 2.0 * spline(0 , m , 4) / delta;
    spline(0 , m , 0) = 3.0 * spline(0 , m , 3) / delta;
  }
}

/* ----------------------------------------------------------------------
   grab n values from file fp and put them in list
   values can be several to a line
   only called by proc 0
------------------------------------------------------------------------- */

void ForceEAM::grab(FILE* fptr, MMD_int n, MMD_float* list)
{
  char* ptr;
  char line[MAXLINE];

  int i = 0;

  while(i < n) {
    fgets(line, MAXLINE, fptr);
    ptr = strtok(line, " \t\n\r\f");
    list[i++] = atof(ptr);

    while(ptr = strtok(NULL, " \t\n\r\f")) list[i++] = atof(ptr);
  }
}

void ForceEAM::communicate(Atom &atom, Comm &comm)
{

  int iswap;
  float_1d_view_type buf;
  MPI_Request request;
  MPI_Status status;

  for(iswap = 0; iswap < comm.nswap; iswap++) {

    /* pack buffer */

    int size = pack_comm(comm.sendnum[iswap], iswap, comm.buf_send, comm.sendlist);

    /* exchange with another proc
       if self, set recv buffer to send buffer */

    if(comm.sendproc[iswap] != me) {
      if(sizeof(MMD_float) == 4) {
        MPI_Irecv(comm.buf_recv.data(), comm.comm_recv_size[iswap], MPI_FLOAT,
                  comm.recvproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(comm.buf_send.data(), comm.comm_send_size[iswap], MPI_FLOAT,
                 comm.sendproc[iswap], 0, MPI_COMM_WORLD);
      } else {
        MPI_Irecv(comm.buf_recv.data(), comm.comm_recv_size[iswap], MPI_DOUBLE,
                  comm.recvproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(comm.buf_send.data(), comm.comm_send_size[iswap], MPI_DOUBLE,
                 comm.sendproc[iswap], 0, MPI_COMM_WORLD);
      }

      MPI_Wait(&request, &status);
      buf = comm.buf_recv;
    } else buf = comm.buf_send;

    /* unpack buffer */

    unpack_comm(comm.recvnum[iswap], comm.firstrecv[iswap], buf);
  }
}
/* ---------------------------------------------------------------------- */

int ForceEAM::pack_comm(int n, int iswap_, float_1d_view_type abuf, const int_2d_lr_view_type& asendlist)
{
  buf = abuf;
  sendlist = asendlist;
  iswap = iswap_;

  Kokkos::parallel_for(Kokkos::RangePolicy<TagEAMPackComm>(0,n),*this);
  /*int i, j, m;

  m = 0;

  for(i = 0; i < n; i++) {
    j = asendlist(iswap,i);
    buf[i] = fp[j];
  }*/

  return 1;
}

KOKKOS_INLINE_FUNCTION
void ForceEAM::operator() (TagEAMPackComm, const int& i) const {
  buf(i) = fp(sendlist(iswap,i));
}

/* ---------------------------------------------------------------------- */

void ForceEAM::unpack_comm(int n, int first_, float_1d_view_type abuf)
{
  buf = abuf;
  first = first_;

  Kokkos::parallel_for(Kokkos::RangePolicy<TagEAMUnpackComm>(0,n),*this);

  /*int i, m, last;

  m = 0;
  last = first + n;

  for(i = first; i < last; i++) fp[i] = buf[m++];*/
}

KOKKOS_INLINE_FUNCTION
void ForceEAM::operator() (TagEAMUnpackComm, const int& i) const {
  fp(first+i) = buf(i);
}

/* ---------------------------------------------------------------------- */

int ForceEAM::pack_reverse_comm(int n, int first, float_1d_view_type buf)
{
  int i, m, last;

  m = 0;
  last = first + n;

  for(i = first; i < last; i++) buf[m++] = rho[i];

  return 1;
}

/* ---------------------------------------------------------------------- */

void ForceEAM::unpack_reverse_comm(int n, int* list, float_1d_view_type buf)
{
  int i, j, m;

  m = 0;

  for(i = 0; i < n; i++) {
    j = list[i];
    rho[j] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

MMD_float ForceEAM::memory_usage()
{
  MMD_int bytes = 2 * nmax * sizeof(MMD_float);
  return bytes;
}


void ForceEAM::bounds(char* str, int nmax, int &nlo, int &nhi)
{
  char* ptr = strchr(str, '*');

  if(ptr == NULL) {
    nlo = nhi = atoi(str);
  } else if(strlen(str) == 1) {
    nlo = 1;
    nhi = nmax;
  } else if(ptr == str) {
    nlo = 1;
    nhi = atoi(ptr + 1);
  } else if(strlen(ptr + 1) == 0) {
    nlo = atoi(str);
    nhi = nmax;
  } else {
    nlo = atoi(str);
    nhi = atoi(ptr + 1);
  }

  if(nlo < 1 || nhi > nmax) printf("Numeric index is out of bounds");
}
