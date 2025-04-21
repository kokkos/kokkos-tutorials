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
  write(*,*)'allocating fortran memory'
  allocate(f_y(mm))

  ! EXERCISE: initialize Kokkos

  ! EXERCISE: allocate Views
  allocate(c_y(mm))
  allocate(x(mm))

  ! put some random numbers in the vectors
  write(*,*)'setting up arrays'
  call random_seed()
  call random_number(f_y)
  do ii = 1,mm
    c_y(ii) = f_y(ii)
  end do
  call random_number(x)
  call random_number(alpha)

  ! perform an axpy in fortran
  write(*,*)'performing an axpy in fortran'
  do ii = 1, mm
    f_y(ii) = f_y(ii) + alpha * x(ii)
  end do

  ! EXERCISE: call the fortran interface to the c_axpy routine

  ! check to see if arrays are "the same"
  if ( norm2(f_y-c_y) < (1.0e-14)*norm2(f_y) ) then
    write(*,*)'PASSED f_y and c_y the same after axpys'
  else
    write(*,*)'FAILED f_y and c_y the same after axpys'
    write(*,*)'norm2(f_y-c_y)',norm2(f_y-c_y)
    write(*,*)'(1.0e-14)*norm2(f_y)',(1.0e-14)*norm2(f_y)
  end if

  ! EXERCISE: deallocate Views

  ! EXERCISE finalize Kokkos

end program example_axpy_view
