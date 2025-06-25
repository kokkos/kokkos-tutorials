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

module axpy_f_mod
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env
  
    use :: flcl_mod
  
    implicit none
  
    public

      interface
        ! Bind the Fortran subroutine f_axpy_view with the C function c_axpy_view
        subroutine f_axpy_view( y, x, alpha ) &
          & bind(c, name='c_axpy_view')
          import
          implicit none
          type(c_ptr), intent(in) :: y
          type(c_ptr), intent(in) :: x
          real(c_double), intent(in) :: alpha
        end subroutine f_axpy_view
      end interface

      contains

        subroutine axpy_view( y, x, alpha )
          ! y and x are one dimensional View of real 64
          type(view_r64_1d_t), intent(inout) :: y
          type(view_r64_1d_t), intent(in) :: x
          real(c_double), intent(in) :: alpha

          call f_axpy_view(y%ptr(), x%ptr(), alpha)

        end subroutine axpy_view

end module axpy_f_mod
