module sympgpr

use fieldlines, only: compute_r

implicit none
save

real(8), parameter :: pi = 4.0d0*datan(1.0d0)

contains

subroutine build_K(x, y, x0, y0, hyp, K)
    ! set up covariance matrix with derivative observations, Eq. (38)
    real(8), external :: d2kdxdx0_num, d2kdxdy0_num, d2kdydx0_num, d2kdydy0_num

    real(8), intent(in) :: x(:), y(:), x0(:), y0(:)
    real(8), intent(in) :: hyp(3)
    real(8), intent(inout) :: K(:,:)

    integer :: i, j, N, N0

    N = size(K, 1)/2
    N0 = size(K, 2)/2

    do j = 1, N0
        do i = 1, N
            K(i,j) = d2kdxdx0_num(&
                x0(j), y0(j), x(i), y(i), hyp(1), hyp(2))
            K(N+i,j) = d2kdxdy0_num(&
                 x0(j), y0(j), x(i), y(i), hyp(1), hyp(2))
            K(i,N0+j) = d2kdxdy0_num(&  ! Symmetry, d2kdydx0 = d2kdxdy0
                 x0(j), y0(j), x(i), y(i), hyp(1), hyp(2))
            K(N+i,N0+j) = d2kdydy0_num(&
                x0(j), y0(j), x(i), y(i), hyp(1), hyp(2))
        end do
    end do
    K = hyp(3)*K
end subroutine build_K

subroutine buildKreg(x, y, x0, y0, hyp, K)
    ! set up "usual" covariance matrix for GP on regular grid (q,p)
    ! print(hyp)
    real(8), external :: kern_num

    real(8), intent(in) :: x(:), y(:), x0(:), y0(:)
    real(8), intent(in) :: hyp(3)
    real(8), intent(inout) :: K(:,:)

    integer :: i, j, N, N0

    N = size(K, 1)
    N0 = size(K, 2)

    do j = 1, N0
        do i = 1, N
            K(i, j) = kern_num(x0(j), y0(j), x(i), y(i), hyp(1), hyp(2))
        end do
    end do
    K = hyp(3)*K
end subroutine buildKreg

function guessP(x, y, hypp, xtrainp, ytrainp, ztrainp, Kyinvp)
    real(8) :: guessP
    real(8), intent(in) :: x(:), y(:)
    real(8), intent(in) :: hypp(3)
    real(8), intent(in) :: xtrainp(:), ytrainp(:), ztrainp(:)
    real(8), intent(in) :: Kyinvp(:,:)

    real(8) :: Kstar(1, size(xtrainp))

    call buildKreg(x, y, xtrainp, ytrainp, hypp, Kstar)
    guessP = dot_product(Kstar(1,:), matmul(Kyinvp, ztrainp))
end function guessP

! def calcQ(x,y, xtrain, l, Kyinv, ztrain):
!     # get \Delta q from GP on mixed grid.
!     Kstar = np.empty((len(xtrain), 2))
!     build_K(xtrain, np.hstack(([x], [y])), l, Kstar)
!     qGP = Kstar.T.dot(Kyinv.dot(ztrain))
!     dq = qGP[1]
!     return dq

! def Pnewton(P, x, y, l, xtrain, Kyinv, ztrain):
!     Kstar = np.empty((len(xtrain), 2))
!     build_K(xtrain, np.hstack((x, P)), l, Kstar)
!     pGP = Kstar.T.dot(Kyinv.dot(ztrain))
!     f = pGP[0] - y + P
!     # print(pGP[0])
!     return f

function calcP(x, y, hyp, hypp, xtrainp, ytrainp, ztrainp, Kyinvp, &
    xtrain, ytrain, ztrain, Kyinv)
    ! as P is given in an implicit relation, use newton to solve for P (Eq.(42))
    ! use the GP on regular grid (q,p) for a first guess for P
    real(8) :: calcP
    real(8), intent(in) :: x(:), y(:)
    real(8), intent(in) :: hyp(3), hypp(3)
    real(8), intent(in) :: xtrainp(:), ytrainp(:), ztrainp(:)
    real(8), intent(in) :: Kyinvp(:,:)
    real(8), intent(in) :: xtrain(:), ytrain(:), ztrain(:)
    real(8), intent(in) :: Kyinv(:,:)

    calcP = guessP(x, y, hypp, xtrainp, ytrainp, ztrainp, Kyinvp)

    ! TODO iterations
    !res, r = newton(Pnewton, pgss, full_output=True, maxiter=50000, disp=True,
    !    args = (np.array([x]), np.array ([y]), l, xtrain, Kyinv, ztrain))
    !return res
end function calcP


! subroutine applymap_tok(nm, Ntest, l, hypp, Q0map, P0map, xtrainp, ztrainp, &
!     Kyinvp, xtrain, ztrain, Kyinv, qmap, pmap)

!     !! Application of symplectic map
!     integer, intent(in) :: nm
!     integer, intent(in) :: Ntest
!     real(8), intent(in) :: l
!     real(8), intent(out) :: pmap(nm, Ntest)
!     real(8), intent(out) :: qmap(nm, Ntest)

!     integer :: i, k
!     real(8) :: zk(3), r, dqmap

!     ! set initial conditions
!     pmap(1,:) = P0map
!     qmap(1,:) = Q0map
!     ! loop through all test points and all time steps
!     do i = 1, nm-1
!         do k = 1, Ntest
!             if ( isnan(pmap(i, k)) ) then
!                 continue
!             else
!                 pmap(i+1, k) = calcP(qmap(i,k), pmap(i,k), l, hypp, xtrainp, &
!                     ztrainp, Kyinvp, xtrain, ztrain, Kyinv, Ntest)

!                 zk(1) = pmap(i+1, k)*1d-2
!                 zk(2) = qmap(i,k)
!                 zk(3) = 0.0d0
!                 r = compute_r(zk, 0.3d0)
!                 if ( (r > 0.5d0) .or. pmap(i+1, k) < 0.0d0 ) then
!                     continue
!                 end if
!             end if
!         end do
!         do k = 1, Ntest
!             if ( isnan(pmap(i+1, k)) ) then
!                 continue
!             else
!                 ! then: set new Q via calculating \Delta q and adding q
!                 dqmap=calcQ(qmap(i,k), pmap(i+1,k), xtrain, l, Kyinv, ztrain)
!                 qmap(i+1, k) = mod(dqmap + qmap(i, k), 2.0d0*pi)
!             end if
!         end do
!     end do
! end subroutine applymap_tok

end module sympgpr
