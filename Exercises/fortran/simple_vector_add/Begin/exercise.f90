! ************************************************************************
!
!                        Kokkos v. 4.0
!       Copyright (2022) National Technology & Engineering
!               Solutions of Sandia, LLC (NTESS).
!
! Under the terms of Contract DE-NA0003525 with NTESS,
! the U.S. Government retains certain rights in this software.
!
! Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
! See https://kokkos.org/LICENSE for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program example_axpy_view
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env

  ! EXERCISE: include flcl
  ! use :: flcl_mod
  ! use :: flcl_util_kokkos_mod
  ! use :: axpy_f_mod

  implicit none

  real(c_double), dimension(:), allocatable :: f_y
  real(REAL64), pointer, dimension(:)  :: c_y
  real(REAL64), pointer, dimension(:)  :: x
  ! EXERCISE: declare Views
  !type(view_r64_1d_t) :: v_c_y
  !type(view_r64_1d_t) :: v_x
  real(c_double) :: alpha
  integer :: mm = 5000                                                                                                                                                                                             
  integer :: ii

  ! allocate fortran memory for array
  print *, 'allocating fortran memory'
  allocate(f_y(mm))

  ! EXERCISE: initialize Kokkos

  ! EXERCISE: allocate Views
  allocate(c_y(mm))
  allocate(x(mm))

  ! put some random numbers in the vectors
  print *, 'setting up arrays'
  call random_seed()
  call random_number(f_y)
  do ii = 1,mm
    c_y(ii) = f_y(ii)
  end do
  call random_number(x)
  call random_number(alpha)

  ! perform an axpy in fortran
  print *, 'performing an axpy in fortran'
  do ii = 1, mm
    f_y(ii) = f_y(ii) + alpha * x(ii)
  end do

  ! EXERCISE: call the fortran interface to the c_axpy routine

  ! check to see if arrays are "the same"
  if ( norm2(f_y-c_y) < (1.0e-14)*norm2(f_y) ) then
    print *, 'PASSED f_y and c_y the same after axpys'
  else
    print *, 'FAILED f_y and c_y the same after axpys'
    print *, 'norm2(f_y-c_y)',norm2(f_y-c_y)
    print *, '(1.0e-14)*norm2(f_y)',(1.0e-14)*norm2(f_y)
  end if

  ! EXERCISE: deallocate Views

  ! EXERCISE finalize Kokkos

end program example_axpy_view
