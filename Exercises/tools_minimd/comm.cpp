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

#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "comm.h"

#define BUFFACTOR 1.5
#define BUFMIN 1000
#define BUFEXTRA 100
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

Comm::Comm()
{
  maxsend = BUFMIN;
  buf_send = float_1d_view_type("Comm::buf_send",maxsend + BUFMIN);
  maxrecv = BUFMIN;
  buf_recv = float_1d_view_type("Comm::buf_send",maxrecv);
  check_safeexchange = 0;
  do_safeexchange = 0;
  maxnlocal = 0;
  count = Kokkos::DualView<int*>("comm::count",1);
}

Comm::~Comm() {}

/* setup spatial-decomposition communication patterns */

int Comm::setup(MMD_float cutneigh, Atom &atom)
{
  int i;
  int periods[3];
  MMD_float prd[3];
  int myloc[3];
  MPI_Comm cartesian;
  MMD_float lo, hi;
  int ineed, idim, nbox;

  prd[0] = atom.box.xprd;
  prd[1] = atom.box.yprd;
  prd[2] = atom.box.zprd;

  /* setup 3-d grid of procs */

  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MMD_float area[3];

  area[0] = prd[0] * prd[1];
  area[1] = prd[0] * prd[2];
  area[2] = prd[1] * prd[2];

  MMD_float bestsurf = 2.0 * (area[0] + area[1] + area[2]);

  // loop thru all possible factorizations of nprocs
  // surf = surface area of a proc sub-domain
  // for 2d, insure ipz = 1

  int ipx, ipy, ipz, nremain;
  MMD_float surf;

  ipx = 1;

  while(ipx <= nprocs) {
    if(nprocs % ipx == 0) {
      nremain = nprocs / ipx;
      ipy = 1;

      while(ipy <= nremain) {
        if(nremain % ipy == 0) {
          ipz = nremain / ipy;
          surf = area[0] / ipx / ipy + area[1] / ipx / ipz + area[2] / ipy / ipz;

          if(surf < bestsurf) {
            bestsurf = surf;
            procgrid[0] = ipx;
            procgrid[1] = ipy;
            procgrid[2] = ipz;
          }
        }

        ipy++;
      }
    }

    ipx++;
  }

  if(procgrid[0]*procgrid[1]*procgrid[2] != nprocs) {
    if(me == 0) printf("ERROR: Bad grid of processors\n");

    return 1;
  }

  /* determine where I am and my neighboring procs in 3d grid of procs */

  int reorder = 0;
  periods[0] = periods[1] = periods[2] = 1;

  MPI_Cart_create(MPI_COMM_WORLD, 3, procgrid, periods, reorder, &cartesian);
  MPI_Cart_get(cartesian, 3, procgrid, periods, myloc);
  MPI_Cart_shift(cartesian, 0, 1, &procneigh[0][0], &procneigh[0][1]);
  MPI_Cart_shift(cartesian, 1, 1, &procneigh[1][0], &procneigh[1][1]);
  MPI_Cart_shift(cartesian, 2, 1, &procneigh[2][0], &procneigh[2][1]);

  /* lo/hi = my local box bounds */

  atom.box.xlo = myloc[0] * prd[0] / procgrid[0];
  atom.box.xhi = (myloc[0] + 1) * prd[0] / procgrid[0];
  atom.box.ylo = myloc[1] * prd[1] / procgrid[1];
  atom.box.yhi = (myloc[1] + 1) * prd[1] / procgrid[1];
  atom.box.zlo = myloc[2] * prd[2] / procgrid[2];
  atom.box.zhi = (myloc[2] + 1) * prd[2] / procgrid[2];

  /* need = # of boxes I need atoms from in each dimension */

  need[0] = static_cast<int>(cutneigh * procgrid[0] / prd[0] + 1);
  need[1] = static_cast<int>(cutneigh * procgrid[1] / prd[1] + 1);
  need[2] = static_cast<int>(cutneigh * procgrid[2] / prd[2] + 1);

  /* alloc comm memory */

  int maxswap = 2 * (need[0] + need[1] + need[2]);

  slablo = float_1d_host_view_type("Comm::slablo",maxswap);
  slabhi = float_1d_host_view_type("Comm::slabhi",maxswap);
  pbc_any = int_1d_host_view_type("Comm::pbc_any",maxswap);
  pbc_flagx = int_1d_host_view_type("Comm::pbc_flagx",maxswap);
  pbc_flagy = int_1d_host_view_type("Comm::pbc_flagy",maxswap);
  pbc_flagz = int_1d_host_view_type("Comm::pbc_flagz",maxswap);
  sendproc = int_1d_host_view_type("Comm::sendproc",maxswap);
  recvproc = int_1d_host_view_type("Comm::recvproc",maxswap);
  sendproc_exc = int_1d_host_view_type("Comm::sendproc_exc",maxswap);
  recvproc_exc = int_1d_host_view_type("Comm::recvproc_exc",maxswap);
  sendnum = int_1d_host_view_type("Comm::sendnum",maxswap);
  recvnum = int_1d_host_view_type("Comm::recvnum",maxswap);
  comm_send_size = int_1d_host_view_type("Comm::comm_send_size",maxswap);
  comm_recv_size = int_1d_host_view_type("Comm::comm_recv_size",maxswap);
  reverse_send_size = int_1d_host_view_type("Comm::reverse_send_size",maxswap);
  reverse_recv_size = int_1d_host_view_type("Comm::reverse_recv_size",maxswap);
  firstrecv = int_1d_host_view_type("Comm::firstrecv",maxswap);
  maxsendlist = int_1d_host_view_type("Comm::maxsendlist",maxswap);

  int iswap = 0;

  for(int idim = 0; idim < 3; idim++)
    for(int i = 1; i <= need[idim]; i++, iswap += 2) {
      MPI_Cart_shift(cartesian, idim, i, &sendproc_exc[iswap], &sendproc_exc[iswap + 1]);
      MPI_Cart_shift(cartesian, idim, i, &recvproc_exc[iswap + 1], &recvproc_exc[iswap]);
    }

  MPI_Comm_free(&cartesian);


  for(i = 0; i < maxswap; i++) maxsendlist[i] = BUFMIN;

  sendlist = int_2d_lr_view_type("Comm::sendlist",maxswap,BUFMIN);

  /* setup 4 parameters for each exchange: (spart,rpart,slablo,slabhi)
     sendproc(nswap) = proc to send to at each swap
     recvproc(nswap) = proc to recv from at each swap
     slablo/slabhi(nswap) = slab boundaries (in correct dimension) of atoms
                            to send at each swap
     1st part of if statement is sending to the west/south/down
     2nd part of if statement is sending to the east/north/up
     nbox = atoms I send originated in this box */

  /* set commflag if atoms are being exchanged across a box boundary
     commflag(idim,nswap) =  0 -> not across a boundary
                          =  1 -> add box-length to position when sending
                          = -1 -> subtract box-length from pos when sending */

  nswap = 0;

  for(idim = 0; idim < 3; idim++) {
    for(ineed = 0; ineed < 2 * need[idim]; ineed++) {
      pbc_any[nswap] = 0;
      pbc_flagx[nswap] = 0;
      pbc_flagy[nswap] = 0;
      pbc_flagz[nswap] = 0;

      if(ineed % 2 == 0) {
        sendproc[nswap] = procneigh[idim][0];
        recvproc[nswap] = procneigh[idim][1];
        nbox = myloc[idim] + ineed / 2;
        lo = nbox * prd[idim] / procgrid[idim];

        if(idim == 0) hi = atom.box.xlo + cutneigh;

        if(idim == 1) hi = atom.box.ylo + cutneigh;

        if(idim == 2) hi = atom.box.zlo + cutneigh;

        hi = MIN(hi, (nbox + 1) * prd[idim] / procgrid[idim]);

        if(myloc[idim] == 0) {
          pbc_any[nswap] = 1;

          if(idim == 0) pbc_flagx[nswap] = 1;

          if(idim == 1) pbc_flagy[nswap] = 1;

          if(idim == 2) pbc_flagz[nswap] = 1;
        }
      } else {
        sendproc[nswap] = procneigh[idim][1];
        recvproc[nswap] = procneigh[idim][0];
        nbox = myloc[idim] - ineed / 2;
        hi = (nbox + 1) * prd[idim] / procgrid[idim];

        if(idim == 0) lo = atom.box.xhi - cutneigh;

        if(idim == 1) lo = atom.box.yhi - cutneigh;

        if(idim == 2) lo = atom.box.zhi - cutneigh;

        lo = MAX(lo, nbox * prd[idim] / procgrid[idim]);

        if(myloc[idim] == procgrid[idim] - 1) {
          pbc_any[nswap] = 1;

          if(idim == 0) pbc_flagx[nswap] = -1;

          if(idim == 1) pbc_flagy[nswap] = -1;

          if(idim == 2) pbc_flagz[nswap] = -1;
        }
      }

      slablo[nswap] = lo;
      slabhi[nswap] = hi;
      nswap++;
    }
  }

  return 0;
}

/* communication of atom info every timestep */

void Comm::communicate(Atom &atom)
{
  int iswap;
  int pbc_flags[4];
  MPI_Request request;
  MPI_Status status;

  for(iswap = 0; iswap < nswap; iswap++) {

    /* pack buffer */

    pbc_flags[0] = pbc_any[iswap];
    pbc_flags[1] = pbc_flagx[iswap];
    pbc_flags[2] = pbc_flagy[iswap];
    pbc_flags[3] = pbc_flagz[iswap];

    int_1d_view_type list = Kokkos::subview(sendlist,iswap,Kokkos::ALL());
    if(sendproc[iswap] != me) {
      atom.pack_comm(sendnum[iswap], list, buf_send, pbc_flags);

    /* exchange with another proc
       if self, set recv buffer to send buffer */

      if(sizeof(MMD_float) == 4) {
        MPI_Irecv(buf_recv.data(), comm_recv_size[iswap], MPI_FLOAT,
        recvproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(buf_send.data(), comm_send_size[iswap], MPI_FLOAT,
        sendproc[iswap], 0, MPI_COMM_WORLD);
      } else {
        MPI_Irecv(buf_recv.data(), comm_recv_size[iswap], MPI_DOUBLE,
        recvproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(buf_send.data(), comm_send_size[iswap], MPI_DOUBLE,
        sendproc[iswap], 0, MPI_COMM_WORLD);
      }

      MPI_Wait(&request, &status);

      buf = buf_recv;
      atom.unpack_comm(recvnum[iswap], firstrecv[iswap], buf);
    } else
      atom.pack_comm_self(sendnum[iswap], list, firstrecv[iswap], pbc_flags);
  }

}

/* reverse communication of atom info every timestep */

void Comm::reverse_communicate(Atom &atom)
{

  int iswap;
  MPI_Request request;
  MPI_Status status;

  for(iswap = nswap - 1; iswap >= 0; iswap--) {

    int_1d_view_type list = Kokkos::subview(sendlist,iswap,Kokkos::ALL());

    /* pack buffer */

    atom.pack_reverse(recvnum[iswap], firstrecv[iswap], buf_send);

    /* exchange with another proc
       if self, set recv buffer to send buffer */

    if(sendproc[iswap] != me) {

      if(sizeof(MMD_float) == 4) {
        MPI_Irecv(buf_recv.data(), reverse_recv_size[iswap], MPI_FLOAT,
        sendproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(buf_send.data(), reverse_send_size[iswap], MPI_FLOAT,
        recvproc[iswap], 0, MPI_COMM_WORLD);
      } else {
        MPI_Irecv(buf_recv.data(), reverse_recv_size[iswap], MPI_DOUBLE,
        sendproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(buf_send.data(), reverse_send_size[iswap], MPI_DOUBLE,
        recvproc[iswap], 0, MPI_COMM_WORLD);
      }
      MPI_Wait(&request, &status);

      buf = buf_recv;
    } else buf = buf_send;

    /* unpack buffer */

    atom.unpack_reverse(sendnum[iswap], list, buf);
  }

}

/* exchange:
   move atoms to correct proc boxes
   send out atoms that have left my box, receive ones entering my box
   this routine called before every reneighboring
   atoms exchanged with all 6 stencil neighbors
*/

void Comm::exchange(Atom &atom_)
{
  atom = atom_;
  int nsend, nrecv, nrecv1, nrecv2, nlocal;

  MPI_Request request;
  MPI_Status status;

  /* enforce PBC */

  atom.pbc();

  /* loop over dimensions */

  for(idim = 0; idim < 3; idim++) {
    /* only exchange if more than one proc in this dimension */

    if(procgrid[idim] == 1) continue;

    /* fill buffer with atoms leaving my box
       when atom is deleted, fill it in with last atom */

    nsend = 0;

    if(idim == 0) {
      lo = atom.box.xlo;
      hi = atom.box.xhi;
    } else if(idim == 1) {
      lo = atom.box.ylo;
      hi = atom.box.yhi;
    } else {
      lo = atom.box.zlo;
      hi = atom.box.zhi;
    }

    x = atom.x;

    nlocal = atom.nlocal;

    if (exc_sendflag.extent(0)<nlocal) {
      Kokkos::resize(exc_sendflag,nlocal);
    }

    count.h_view(0) = exc_sendlist.extent(0);

    while (count.h_view(0)>=exc_sendlist.extent(0)) {
      count.h_view(0) = 0;
      count.modify<HostType>();
      count.sync<DeviceType>();

      Kokkos::parallel_for("Comm::exchange_sendlist",Kokkos::RangePolicy<TagExchangeSendlist>(0,nlocal),*this);
      Kokkos::fence();

      count.modify<DeviceType>();
      count.sync<HostType>();
      if ((count.h_view(0)>=exc_sendlist.extent(0)) ||
          (count.h_view(0)>=exc_copylist.extent(0)) ) {
        Kokkos::resize(exc_sendlist,(count.h_view(0)+1)*1.1);
        Kokkos::resize(exc_copylist,(count.h_view(0)+1)*1.1);
        count.h_view(0)=exc_sendlist.extent(0);
      }
      if (count.h_view(0)*7>=maxsend)
        growsend(count.h_view(0));
    }
    h_exc_sendflag = Kokkos::create_mirror_view(exc_sendflag);
    h_exc_copylist = Kokkos::create_mirror_view(exc_copylist);
    h_exc_sendlist = Kokkos::create_mirror_view(exc_sendlist);

    Kokkos::deep_copy(h_exc_sendflag,exc_sendflag);
    Kokkos::deep_copy(h_exc_copylist,exc_copylist);
    Kokkos::deep_copy(h_exc_sendlist,exc_sendlist);

    int sendpos = nlocal-1;
    nlocal -= count.h_view(0);
    for(int i = 0; i < count.h_view(0); i++) {
      if (h_exc_sendlist(i)<nlocal) {
        while (h_exc_sendflag(sendpos)) sendpos--;
        h_exc_copylist(i) = sendpos;
        sendpos--;
      } else
        h_exc_copylist(i) = -1;
    }
    Kokkos::deep_copy(exc_copylist,h_exc_copylist);

    Kokkos::parallel_for("Comm::exchange_pack",Kokkos::RangePolicy<TagExchangePack>(0,count.h_view(0)),*this);

    atom.nlocal -= count.h_view(0);
    Kokkos::fence();

    nsend = count.h_view(0) * 7;



      MPI_Send(&nsend, 1, MPI_INT, procneigh[idim][0], 0, MPI_COMM_WORLD);
      MPI_Recv(&nrecv1, 1, MPI_INT, procneigh[idim][1], 0, MPI_COMM_WORLD, &status);
      nrecv = nrecv1;

      if(procgrid[idim] > 2) {
        MPI_Send(&nsend, 1, MPI_INT, procneigh[idim][1], 0, MPI_COMM_WORLD);
        MPI_Recv(&nrecv2, 1, MPI_INT, procneigh[idim][0], 0, MPI_COMM_WORLD, &status);
        nrecv += nrecv2;
      }

      if(nrecv > maxrecv) growrecv(nrecv);

      if(sizeof(MMD_float) == 4) {
        MPI_Irecv(buf_recv.data(), nrecv1, MPI_FLOAT, procneigh[idim][1], 0,
                  MPI_COMM_WORLD, &request);
        MPI_Send(buf_send.data(), nsend, MPI_FLOAT, procneigh[idim][0], 0, MPI_COMM_WORLD);
      } else {
        MPI_Irecv(buf_recv.data(), nrecv1, MPI_DOUBLE, procneigh[idim][1], 0,
                  MPI_COMM_WORLD, &request);
        MPI_Send(buf_send.data(), nsend, MPI_DOUBLE, procneigh[idim][0], 0, MPI_COMM_WORLD);
      }

      MPI_Wait(&request, &status);

      if(procgrid[idim] > 2) {
        if(sizeof(MMD_float) == 4) {
          MPI_Irecv(buf_recv.data()+nrecv1, nrecv2, MPI_FLOAT, procneigh[idim][0], 0,
                    MPI_COMM_WORLD, &request);
          MPI_Send(buf_send.data(), nsend, MPI_FLOAT, procneigh[idim][1], 0, MPI_COMM_WORLD);
        } else {
          MPI_Irecv(buf_recv.data()+nrecv1, nrecv2, MPI_DOUBLE, procneigh[idim][0], 0,
                    MPI_COMM_WORLD, &request);
          MPI_Send(buf_send.data(), nsend, MPI_DOUBLE, procneigh[idim][1], 0, MPI_COMM_WORLD);
        }

        MPI_Wait(&request, &status);
      }

      nrecv_atoms = nrecv / 7;

    /* check incoming atoms to see if they are in my box
       if they are, add to my list */

    nrecv = 0;

    Kokkos::parallel_reduce("Comm::exchange_count_recv",Kokkos::RangePolicy<TagExchangeCountRecv>(0,nrecv_atoms),*this,nrecv);

    nlocal = atom.nlocal;

    if(nrecv_atoms>0)
    atom.nlocal += nrecv;

    count.h_view(0) = nlocal;
    count.modify<HostType>();
    count.sync<DeviceType>();

    if(atom.nlocal>=atom.nmax)
      atom.growarray();

    Kokkos::parallel_for("Comm::exchange_unpack",Kokkos::RangePolicy<TagExchangeUnpack>(0,nrecv_atoms),*this);

  }
  atom_ = atom;
}

KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagExchangeSendlist, const int& i) const {
  if (x(i,idim) < lo || x(i,idim) >= hi) {
    const int mysend=Kokkos::atomic_fetch_add(&count.d_view(0),1);
    if(mysend<exc_sendlist.extent(0)) {
      exc_sendlist(mysend) = i;
      exc_sendflag(i) = 1;
    }
  } else
    exc_sendflag(i) = 0;
}
KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagExchangePack, const int& i ) const {
  atom.pack_exchange(exc_sendlist(i),&buf_send[7*i]);

  if(exc_copylist(i) > 0)
    atom.copy(exc_copylist(i),exc_sendlist(i));
}
KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagExchangeCountRecv, const int& i, int& sum) const {
  const MMD_float value = buf_recv[i * 7 + idim];
  if(value >= lo && value < hi)
    sum++;
}
KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagExchangeUnpack, const int& i ) const {
  double value = buf_recv[i * 7 + idim];

  if(value >= lo && value < hi)
    atom.unpack_exchange(Kokkos::atomic_fetch_add(&count.d_view(0),1), &buf_recv[i * 7]);
}

/* borders:
   make lists of nearby atoms to send to neighboring procs at every timestep
   one list is created for every swap that will be made
   as list is made, actually do swaps
   this does equivalent of a communicate (so don't need to explicitly
     call communicate routine on reneighboring timestep)
   this routine is called before every reneighboring
*/

void Comm::borders(Atom &atom_)
{
  atom = atom_;
  int ineed, nsend, nrecv, nfirst, nlast;
  MPI_Request request;
  MPI_Status status;

  /* erase all ghost atoms */

  atom.nghost = 0;

  /* do swaps over all 3 dimensions */

  iswap = 0;


  if(atom.nlocal > maxnlocal) {
    send_flag = int_1d_view_type("Comm::sendflag",atom.nlocal);
    maxnlocal = atom.nlocal;
  }

  for(idim = 0; idim < 3; idim++) {
    nlast = 0;

    for(ineed = 0; ineed < 2 * need[idim]; ineed++) {

      // find atoms within slab boundaries lo/hi using <= and >=
      // check atoms between nfirst and nlast
      //   for first swaps in a dim, check owned and ghost
      //   for later swaps in a dim, only check newly arrived ghosts
      // store sent atom indices in list for use in future timesteps

      lo = slablo[iswap];
      hi = slabhi[iswap];
      pbc_flags[0] = pbc_any[iswap];
      pbc_flags[1] = pbc_flagx[iswap];
      pbc_flags[2] = pbc_flagy[iswap];
      pbc_flags[3] = pbc_flagz[iswap];

      x = atom.x;

      if(ineed % 2 == 0) {
        nfirst = nlast;
        nlast = atom.nlocal + atom.nghost;
      }

      nsend = 0;

      count.h_view(0) = 0;
      count.modify<HostType>();
      count.sync<DeviceType>();

      send_count = count.d_view;

      Kokkos::parallel_for("Comm::border_sendlist",Kokkos::RangePolicy<TagBorderSendlist>(nfirst,nlast),*this);

      count.modify<DeviceType>();
      count.sync<HostType>();

      nsend = count.h_view(0);
      if(nsend > exc_sendlist.extent(0)) {
        Kokkos::resize(exc_sendlist , nsend);

        growlist(iswap, nsend);

        count.h_view(0) = 0;
        count.modify<HostType>();
        count.sync<DeviceType>();

        Kokkos::parallel_for("Comm::border_sendlist",Kokkos::RangePolicy<TagBorderSendlist>(nfirst,nlast),*this);

        count.modify<DeviceType>();
        count.sync<HostType>();
      }

      if(nsend * 4 > maxsend) growsend(nsend * 4);

      Kokkos::parallel_for("Comm::border_pack",Kokkos::RangePolicy<TagBorderPack>(0,nsend),*this);

      /* swap atoms with other proc
      put incoming ghosts at end of my atom arrays
      if swapping with self, simply copy, no messages */


        if(sendproc[iswap] != me) {
          MPI_Send(&nsend, 1, MPI_INT, sendproc[iswap], 0, MPI_COMM_WORLD);
          MPI_Recv(&nrecv, 1, MPI_INT, recvproc[iswap], 0, MPI_COMM_WORLD, &status);

          if(nrecv * atom.border_size > maxrecv) growrecv(nrecv * atom.border_size);

          if(sizeof(MMD_float) == 4) {
            MPI_Irecv(buf_recv.data(), nrecv * atom.border_size, MPI_FLOAT,
                      recvproc[iswap], 0, MPI_COMM_WORLD, &request);
            MPI_Send(buf_send.data(), nsend * atom.border_size, MPI_FLOAT,
                     sendproc[iswap], 0, MPI_COMM_WORLD);
          } else {
            MPI_Irecv(buf_recv.data(), nrecv * atom.border_size, MPI_DOUBLE,
                      recvproc[iswap], 0, MPI_COMM_WORLD, &request);
            MPI_Send(buf_send.data(), nsend * atom.border_size, MPI_DOUBLE,
                     sendproc[iswap], 0, MPI_COMM_WORLD);
          }

          MPI_Wait(&request, &status);
          buf = buf_recv;
        } else {
          nrecv = nsend;
          buf = buf_send;
        }

      /* unpack buffer */

      n = atom.nlocal + atom.nghost;

      while(n + nrecv > atom.nmax) atom.growarray();

      x = atom.x;

      Kokkos::parallel_for("Comm::border_unpack",Kokkos::RangePolicy<TagBorderUnpack>(0,nrecv),*this);

      /* set all pointers & counters */

        sendnum[iswap] = nsend;
        recvnum[iswap] = nrecv;
        comm_send_size[iswap] = nsend * atom.comm_size;
        comm_recv_size[iswap] = nrecv * atom.comm_size;
        reverse_send_size[iswap] = nrecv * atom.reverse_size;
        reverse_recv_size[iswap] = nsend * atom.reverse_size;
        firstrecv[iswap] = atom.nlocal + atom.nghost;
        atom.nghost += nrecv;

      iswap++;
    }
  }

  /* insure buffers are large enough for reverse comm */

  int max1, max2;
  max1 = max2 = 0;

  for(iswap = 0; iswap < nswap; iswap++) {
    max1 = MAX(max1, reverse_send_size[iswap]);
    max2 = MAX(max2, reverse_recv_size[iswap]);
  }

  if(max1 > maxsend) growsend(max1);

  if(max2 > maxrecv) growrecv(max2);
  atom_ = atom;

}

KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagBorderSendlist, const int& i) const {
  if(x(i,idim) >= lo && x(i,idim) <= hi) {
    const int nsend = (send_count(0)+=1)-1;
    if(nsend < exc_sendlist.extent(0)) {
      exc_sendlist[nsend] = i;
    }
  }
}

KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagBorderPack, const int& k) const {
  atom.pack_border(exc_sendlist(k), &buf_send[k * 4], pbc_flags);
  sendlist(iswap,k) = exc_sendlist(k);
}

KOKKOS_INLINE_FUNCTION
void Comm::operator() (TagBorderUnpack, const int& i) const {
  atom.unpack_border(n + i, &buf[i * 4]);
}

/* realloc the size of the send buffer as needed with BUFFACTOR & BUFEXTRA */

void Comm::growsend(int n)
{
  Kokkos::resize(buf_send,static_cast<int>(BUFFACTOR * n) + BUFEXTRA);
  maxsend = static_cast<int>(BUFFACTOR * n);
}

/* free/malloc the size of the recv buffer as needed with BUFFACTOR */

void Comm::growrecv(int n)
{
  maxrecv = static_cast<int>(BUFFACTOR * n) + BUFEXTRA;
  buf_recv = float_1d_view_type("Comm::buf_send",maxrecv);;
}

/* realloc the size of the iswap sendlist as needed with BUFFACTOR */

void Comm::growlist(int iswap, int n)
{
  if(n<=maxsendlist[iswap]) return;
  int maxswap = sendlist.extent(0);
  Kokkos::resize(sendlist,sendlist.extent(0),BUFFACTOR * n + BUFEXTRA);
  for(int iswaps = 0; iswaps < maxswap; iswaps++) {
    maxsendlist[iswaps] = static_cast<int>(BUFFACTOR * n);
  }
}
