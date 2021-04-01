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

function calcQ(x, y, xtrain, ytrain, hyp, Kyinv, ztrain)
    ! get \Delta q from GP on mixed grid.
    real(8) :: calcQ
    real(8), intent(in) :: x(:), y(:)
    real(8), intent(in) :: hyp(3)
    real(8), intent(in) :: xtrain(:), ytrain(:), ztrain(:)
    real(8), intent(in) :: Kyinv(:,:)

    real(8) :: Kstar(2, 2*size(xtrain))
    call build_K(x, y, xtrain, ytrain, hyp, Kstar)
    calcQ = dot_product(Kstar(2, :), matmul(Kyinv, ztrain))
end function calcQ

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

    integer :: info
    real(8) :: pgss(1), fvec(1)

    pgss(1) = guessP(x, y, hypp, xtrainp, ytrainp, ztrainp, Kyinvp)
    call target(1, pgss, fvec, info)

    ! pgss = 1.08172922d0
    call hybrd1(target, 1, pgss, fvec, 1d-13, info)
    calcP = pgss(1)

    contains

    subroutine target(n, p, f, iflag)
        integer, intent(in) :: n
        real(8), intent(in) :: p(n)
        real(8), intent(out) :: f(n)
        integer, intent(in) :: iflag

        real(8) :: pGP
        real(8) :: Kstar(2, 2*size(xtrain))
        call build_K(x, p, xtrain, ytrain, hyp, Kstar)
        pGP = dot_product(Kstar(1, :), matmul(Kyinv, ztrain))
        f(1) = pGP - y(1) + p(1)

    end subroutine target
end function calcP


subroutine applymap_tok(nm, Ntest, hyp, hypp, Q0map, P0map, xtrainp, ytrainp, &
    ztrainp, Kyinvp, xtrain, ytrain, ztrain, Kyinv, qmap, pmap)

    !! Application of symplectic map
    integer :: nm
    integer :: Ntest
    real(8), intent(in) :: hyp(3), hypp(3)
    real(8), intent(in) :: Q0map(Ntest), P0map(Ntest)
    real(8), intent(in) :: xtrainp(:), ytrainp(:), ztrainp(:), Kyinvp(:, :)
    real(8), intent(in) :: xtrain(:), ytrain(:), ztrain(:), Kyinv(:, :)
    real(8), intent(inout) :: pmap(nm, Ntest, 1)
    real(8), intent(inout) :: qmap(nm, Ntest, 1)

    integer :: i, k
    real(8) :: zk(3), r, dqmap

    ! set initial conditions
    pmap(1,:,1) = P0map
    qmap(1,:,1) = Q0map
    ! loop through all test points and all time steps
    do i = 1, nm-1
        do k = 1, Ntest
            if ( isnan(pmap(i, k, 1)) ) then
                continue
            else
                pmap(i+1, k, 1) = calcP(qmap(i,k,:), pmap(i,k,:), hyp, hypp, &
                    xtrainp, ytrainp, ztrainp, Kyinvp, xtrain, ytrain, ztrain, &
                    Kyinv)

                zk(1) = pmap(i+1, k, 1)*1d-2
                zk(2) = qmap(i,k, 1)
                zk(3) = 0.0d0
                r = compute_r(zk, 0.3d0)
                if ( (r > 0.5d0) .or. pmap(i+1, k, 1) < 0.0d0 ) then
                    continue
                end if
            end if
        end do
        do k = 1, Ntest
            if ( isnan(pmap(i+1, k, 1)) ) then
                continue
            else
                ! then: set new Q via calculating \Delta q and adding q
                dqmap=calcQ(qmap(i,k,:), pmap(i+1,k,:), xtrain, ytrain, hyp, &
                    Kyinv, ztrain)
                qmap(i+1, k, 1) = mod(dqmap + qmap(i, k, 1), 2.0d0*pi)
            end if
        end do
    end do
end subroutine applymap_tok

end module sympgpr
