module henon
  implicit none
  save

  integer, parameter :: n = 4        ! Number of phase-space dimensions
  integer, parameter :: ncut = 5024  ! Maximum number of cuts

  real(8) :: E_bound = 1d0/12d0
  real(8) :: lam = 1d0
  real(8) :: w1 = 1d0
  real(8) :: w2 = 1d0

  real(8) :: tmacro = 1d0            ! Macrosteps for integration
  real(8) :: tmax = 200d0             ! Maximum integration time
end
