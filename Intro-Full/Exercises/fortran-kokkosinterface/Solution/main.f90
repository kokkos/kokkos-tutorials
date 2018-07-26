program main

  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env
  use :: abi_mod
  use :: f_interface_mod

  implicit none

  integer :: n
  real(c_double) :: alpha
  real(c_double), dimension(:), allocatable :: array_x
  real(c_double), dimension(:), allocatable :: f_array_y, c_array_y

  n = 20
  allocate( array_x(n) )
  allocate( c_array_y(n) )
  allocate( f_array_y(n) )

  alpha = 0.5
  array_x = 1
  f_array_y = 1
  c_array_y = 1

  ! f axpy
  f_array_y = alpha * array_x + f_array_y

  ! alpha = 2.0
  ! c_axpy
  call axpy_kokkos( alpha, array_x, c_array_y )

  if ( abs(sum(f_array_y) - sum(c_array_y)) .le. 1.0**(-15) ) then
    write(*,*)'Good job!'
  else
    write(*,*)'Please try again.'
  end if

end program main
