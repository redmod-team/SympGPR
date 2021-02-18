program main
  use henon, only: n, ncut
  implicit none

  real(8) :: z0(n)  ! Initial conditions
  real(8) :: tcut(ncut), zcut(n, ncut)  ! Cut time and phase-position
  integer :: icut  ! Cut count

  z0(1) = 0d0
  z0(2) = 0.2d0
  z0(3) = 0.1d0
  z0(4) = 0.1d0

  call integrate(z0, tcut, zcut, icut)
end program main


subroutine tstep(n, t, z, dzdt)
  use henon, only: w1, w2, lam  ! read-only
  implicit none

  integer, intent(in) :: n
  real(8), intent(in) :: t
  real(8), intent(in) :: z(n)
  real(8), intent(out) :: dzdt(n)

  dzdt(3) = -w1 * z(1) - 2*lam*z(1)*z(2)
  dzdt(4) = -w2 * z(2) - lam*(z(1)**2 - z(2)**2)
  dzdt(1) = w1 * z(3)
  dzdt(2) = w2 * z(4)
end subroutine tstep


subroutine fcut(n, t, z, ng, gout)
  implicit none
! For finding roots for Poincare cuts

  integer, intent(in) :: n, ng
  real(8), intent(in) :: t, z(n)
  real(8), intent(out) :: gout(ng)

  gout(1) = z(1)  ! Find cut z(1) == 0
end subroutine fcut


subroutine integrate(z0, tcut, zcut, icut)
  use henon, only: n, ncut, tmax, tmacro  ! read-only
  use dvode_f90_m, only: vode_opts, set_normal_opts, dvode_f90
  implicit none

  external tstep, fcut

  real(8), intent(in)  :: z0(n)     ! Initial conditions
  real(8), intent(out) :: tcut(ncut), zcut(n, ncut)  ! Saving cuts

  integer, intent(out) :: icut     ! Cut counter
  real(8) :: t1, t2, z(n)          ! Current timestep

  ! Options for VODE
  real(8) :: atol(n), rtol, rstats(22)
  integer(4) :: itask, istate, istats(31), nevents
  type (vode_opts) :: options

  ! Set VODE options
  rtol = 1d-12
  atol = 1d-13
  itask = 1
  istate = 1
  nevents = 1  ! One event for z(1) == 0
  options = set_normal_opts(abserr_vector=atol, relerr=rtol, nevents=nevents)

  z = z0
  icut = 0

  do
    t1 = t2
    t2 = t1 + tmacro
    ! g_fcn specifies events
    call dvode_f90(tstep, n, z, t1, t2, itask, istate, options, g_fcn = fcut)

    if (istate == 3 .and. z(3) > 0) then
      icut = icut + 1
      if (icut > ncut) exit
      tcut(icut) = t2
      zcut(:, icut) = z
    end if
    if (t2 > tmax) exit
  end do
end subroutine integrate
