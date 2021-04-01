import numpy as np
from scipy.optimize import newton, bisect

from func import f_kern, d2kdxdx0, d2kdxdy0, d2kdydx0, d2kdydy0
from fieldlines import fieldlines

def build_K(xin, x0in, hyp, K):
    # set up covariance matrix with derivative observations, Eq. (38)
    l = hyp[:-1]
    sig = hyp[-1]
    N = K.shape[0]//2
    N0 = K.shape[1]//2
    x0 = x0in[0:N0]
    x = xin[0:N]
    y0 = x0in[N0:2*N0]
    y = xin[N:2*N]
    for k in range(N):
        for lk in range(N0):
            K[k,lk] = d2kdxdx0(
                x0[lk], y0[lk], x[k], y[k], l)
            K[N+k,lk] = d2kdxdy0(
                 x0[lk], y0[lk], x[k], y[k], l)
            K[k,N0+lk] = d2kdydx0(
                 x0[lk], y0[lk], x[k], y[k], l)
            K[N+k,N0+lk] = d2kdydy0(
                x0[lk], y0[lk], x[k], y[k], l)
    K[:,:] = sig*K[:,:]

def buildKreg(xin, x0in, hyp, K):
    # set up "usual" covariance matrix for GP on regular grid (q,p)
    # print(hyp)
    l = hyp[:-1]
    sig = hyp[-1]
    N = K.shape[0]
    N0 = K.shape[1]
    x0 = x0in[0:N0]
    x = xin[0:N]
    y0 = x0in[N0:2*N0]
    y = xin[N:2*N]
    for k in range(N):
        for lk in range(N0):
            K[k,lk] = f_kern(x0[lk], y0[lk], x[k], y[k], l)
    K[:,:] = sig*K[:,:]

def guessP(x, y, hypp, xtrainp, ztrainp, Kyinvp):
    Kstar = np.empty((1, int(len(xtrainp)/2)))
    buildKreg(np.hstack((x,y)), xtrainp, hypp, Kstar)
    Ef = Kstar.dot(Kyinvp.dot(ztrainp))
    return Ef

def calcQ(x,y, xtrain, l, Kyinv, ztrain):
    # get \Delta q from GP on mixed grid.
    Kstar = np.empty((len(xtrain), 2), order='F')
    build_K(xtrain, np.hstack(([x], [y])), l, Kstar)
    qGP = Kstar.T.dot(Kyinv.dot(ztrain))
    dq = qGP[1]
    return dq

def Pnewton(P, x, y, l, xtrain, Kyinv, ztrain):
    Kstar = np.empty((len(xtrain), 2), order='F')
    build_K(xtrain, np.hstack((x, P)), l, Kstar)
    pGP = Kstar.T.dot(Kyinv.dot(ztrain))
    f = pGP[0] - y + P
    return f

def calcP(x,y, l, hypp, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv):
    # as P is given in an implicit relation, use newton to solve for P (Eq.(42))
    # use the GP on regular grid (q,p) for a first guess for P
    pgss = guessP([x], [y], hypp, xtrainp, ztrainp, Kyinvp)
    res, r = newton(Pnewton, pgss, full_output=True, maxiter=50000, disp=True,
        args = (np.array([x]), np.array([y]), l, xtrain, Kyinv, ztrain))
    return res

def applymap_tok(nm, Ntest, l, hypp, Q0map, P0map, xtrainp, ztrainp, Kyinvp,
    xtrain, ztrain, Kyinv):
    # Application of symplectic map
    #init
    pmap = np.zeros([nm, Ntest])
    qmap = np.zeros([nm, Ntest])
    #set initial conditions
    pmap[0,:] = P0map
    qmap[0,:] = Q0map
    # loop through all test points and all time steps
    for i in range(0,nm-1):
        for k in range(0, Ntest):
            if np.isnan(pmap[i, k]):
                pmap[i+1,k] = np.nan
            else:
                pmap[i+1, k] = calcP(qmap[i,k], pmap[i, k], l, hypp, xtrainp,
                    ztrainp, Kyinvp, xtrain, ztrain, Kyinv)

                zk = np.array([pmap[i+1, k]*1e-2, qmap[i,k], 0])
                temp = fieldlines.compute_r(zk, 0.3)
                if temp > 0.5 or pmap[i+1, k] < 0.0:
                    pmap[i+1, k] = np.nan
        for k in range(0, Ntest):
            if np.isnan(pmap[i+1, k]):
                qmap[i+1,k] = np.nan
            else:
                # then: set new Q via calculating \Delta q and adding q
                dqmap = calcQ(qmap[i,k], pmap[i+1,k], xtrain, l, Kyinv, ztrain)
                qmap[i+1, k] = np.mod(dqmap + qmap[i, k], 2.0*np.pi)
    return qmap, pmap
