module fieldlines
implicit none
save

real(8), parameter :: pi = 4d0*atan(1d0)

real(8), parameter :: B0 = 1d0
real(8), parameter :: iota0 = 1d0 ! constant part of rotational transform
real(8), parameter :: a = 0.5d0   ! (equivalent) minor radius
real(8), parameter :: R0 = 1d0    ! (equivalent) major radius

real(8) :: dph      ! Timestep
real(8) :: eps      ! Absolute perturbation
real(8) :: phase    ! Perturbation phase
integer :: m, n     ! Perturbation harmonics

real(8) :: rlast = 0d0

contains

subroutine init(nph, am, an, aeps, aphase, arlast)
    integer :: nph, am, an
    real(8) :: aeps, aphase, arlast

    dph = 2*pi/nph
    m = am
    n = an
    eps = aeps
    phase = aphase
    rlast = arlast
end subroutine init


function Ath(r, th, ph)
    real(8), intent(in) :: r, th, ph
    real(8) :: Ath

    Ath = B0*(r**2/2d0 - r**3/(3d0*R0)*cos(th))
end function Ath


function dAthdr(r, th, ph)
    real(8), intent(in) :: r, th, ph
    real(8) :: dAthdr

    dAthdr = B0*(r - r**2/R0*cos(th))
end function dAthdr


function dAthdth(r, th, ph)
    real(8), intent(in) :: r, th, ph
    real(8) :: dAthdth

    dAthdth = B0*r**3*sin(th)/(3d0*R0)
end function dAthdth


function Aph(r, th, ph)
    real(8), intent(in) :: r, th, ph
    real(8) :: Aph

    Aph = -B0*iota0*(r**2/2d0-r**4/(4d0*a**2))*(1d0+eps*cos(m*th+n*ph+phase))
end function Aph


function dAphdr(r, th, ph)
    real(8), intent(in) :: r, th, ph
    real(8) :: dAphdr

    dAphdr = -B0*iota0*(r-r**3/a**2)*(1d0+eps*cos(m*th+n*ph+phase))
end function dAphdr


function dAphdth(r, th, ph)
    real(8), intent(in) :: r, th, ph
    real(8) :: dAphdth

    dAphdth = B0*iota0*(r**2/2d0-r**4/(4d0*a**2))*m*eps*sin(m*th + n*ph + phase)
end function dAphdth


subroutine f_r(x, y, dy, args)
    ! Implicit function to solve for r
    real(8), intent(in) :: x        ! r
    real(8), intent(in) :: args(3)  ! pth, th, ph
    real(8), intent(out) :: y   ! Target function
    real(8), intent(out) :: dy  ! Jacobian
    ! Compute r implicitly for given th, p_th and ph
    y = args(1) - Ath(x, args(2), args(3))  ! pth - pth(r, th, ph)
    dy = -dAthdr(x, args(2), args(3))
end subroutine f_r


function compute_r(z, rstart) result(r)
    real(8), intent(in) :: z(3)
    real(8), intent(in) :: rstart
    real(8) :: r

    integer :: k
    real(8) :: y, dy

    r = rstart
    do k = 1, 20  ! Newton iterations
        call f_r(r, y, dy, z)
        r = r - y/dy
    end do
end function compute_r


subroutine F_tstep(znew, y, dy, zold)
    real(8), intent(in) :: znew(2)
    real(8), intent(in) :: zold(3)
    real(8), intent(out) :: y(2)     ! Target function
    real(8), intent(out) :: dy(2,2)  ! Jacobian

    real(8) :: z(3), r
    real(8) :: dApdr, dApdt, dAtdr, dAtdt

    z(1:2) = 0.5d0*(zold(1:2) + znew)
    z(3) = zold(3) + 0.5d0*dph
    r = compute_r(z, rlast)
    rlast = r

    dApdr = dAphdr(r, z(2), z(3))
    dApdt = dAphdth(r, z(2), z(3))
    dAtdr = dAthdr(r, z(2), z(3))
    dAtdt = dAthdth(r, z(2), z(3))

    y(1) = zold(1) - znew(1) + dph*(dApdt - dApdr*dAtdt/dAtdr)
    y(2) = zold(2) - znew(2) - dph*dApdr/dAtdr

    ! print *, y

    ! TODO: look in SIMPLE
    dy(1,1) = 0d0
    dy(2,1) = 0d0
    dy(1,2) = 0d0
    dy(2,2) = 0d0

end subroutine F_tstep


subroutine timestep(z)
    real(8), intent(inout) :: z(3)

    integer :: info
    real(8) :: zold(3)
    real(8) :: z12new(2)
    real(8) :: fvec(3)

    zold = z
    z12new = z(1:2)

    call hybrd1(f_tstep_wrap, 2, z12new, fvec, 1d-13, info)
!    print *, info

    z(1:2) = z12new
    z(3) = zold(3) + dph

    contains

    subroutine f_tstep_wrap(n, x, fvec, iflag)
        integer, intent(in) :: n
        double precision, intent(in) :: x(n)
        double precision, intent(out) :: fvec(n)
        integer, intent(in) :: iflag

        real(8) :: dummy(2,2)  ! for later Jacobian

        call F_tstep(x, fvec, dummy, zold)
    end subroutine f_tstep_wrap
end subroutine timestep

end module fieldlines
