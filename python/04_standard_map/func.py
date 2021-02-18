# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:47:00 2020

@author: Katharina Rath
"""

import numpy as np
from scipy.optimize import newton, bisect
from scipy.linalg import solve_triangular
import scipy
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp

from kernels import * # implicit SympGPR
#from kernels_expl_per_q_sq_p import * #explicit SympGPR
def f_kern(x, y, x0, y0, l):
    return kern_num(x,y,x0,y0,l[0], l[1])

def d2kdxdx0(x, y, x0, y0, l):
    return d2kdxdx0_num(x,y,x0,y0,l[0], l[1])

def d2kdydy0(x, y, x0, y0, l):
    return d2kdydy0_num(x,y,x0,y0,l[0], l[1])

def d2kdxdy0(x, y, x0, y0, l):
    return d2kdxdy0_num(x,y,x0,y0,l[0], l[1])

def d2kdydx0(x, y, x0, y0, l):
    return d2kdxdy0(x, y, x0, y0, l)

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
            K[k,lk] = f_kern(
                     x0[lk], y0[lk], x[k], y[k], l)
    K[:,:] = sig*K[:,:]


def gpsolve(Ky, ft):
    L = scipy.linalg.cholesky(Ky, lower = True)
    alpha = solve_triangular(
        L.T, solve_triangular(L, ft, lower=True, check_finite=False),
        lower=False, check_finite=False)

    return L, alpha

# compute log-likelihood according to RW, p.19
def solve_cholesky(L, b):
    return solve_triangular(
        L.T, solve_triangular(L, b, lower=True, check_finite=False),
        lower=False, check_finite=False)

def nll_chol_reg(hyp, x, y, N):
    K = np.empty((N, N))
    buildKreg(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
    L = scipy.linalg.cholesky(Ky, lower = True)
    alpha = solve_cholesky(L, y)
    ret = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))
    return ret
# negative log-posterior
def nll_chol(hyp, x, y, N):
    K = np.empty((N, N))
    build_K(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
    L = scipy.linalg.cholesky(Ky, lower = True)
    alpha = solve_cholesky(L, y)
    ret = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))
    return ret

def build_K_expl(xin, x0in, hyp, K):
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
    
def nll_expl(hyp, x, y, N, ind):
    K = np.empty((N, N))
    if ind == 0:
        build_K_expl(x, x, np.hstack((hyp[0], 0, hyp[1])), K)
        Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
        Ky = Ky[0:len(y), 0:len(y)]
    else:
        build_K_expl(x, x, np.hstack((0, hyp[0], hyp[1])), K)
        Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
        Ky = Ky[len(y):2*len(y), len(y):2*len(y)]

    L = scipy.linalg.cholesky(Ky, lower = True)
    alpha = solve_cholesky(L, y)
    nlp_val = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))

    return nlp_val

def guessP(x, y, hypp, xtrainp, ztrainp, Kyinvp, N):
    Ntest = 1
    Kstar = np.empty((Ntest, int(len(xtrainp)/2)))
    buildKreg(np.hstack((x,y)), xtrainp, hypp, Kstar)
    Ef = Kstar.dot(Kyinvp.dot(ztrainp))
    return Ef

def calcQ(x,y, xtrain, l, Kyinv, ztrain):
    # get \Delta q from GP on mixed grid.
    Kstar = np.empty((len(xtrain), 2))
    build_K(xtrain, np.hstack(([x], [y])), l, Kstar)
    qGP = Kstar.T.dot(Kyinv.dot(ztrain))
    dq = qGP[1]
    return dq

def Pnewton(P, x, y, l, xtrain, Kyinv, ztrain):
    Kstar = np.empty((len(xtrain), 2))
    build_K(xtrain, np.hstack((x, P)), l, Kstar)
    pGP = Kstar.T.dot(Kyinv.dot(ztrain))
    f = pGP[0] - y + P
    # print(pGP[0])
    return f

def calcP(x,y, l, hypp, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv, Ntest):
    # as P is given in an implicit relation, use newton to solve for P (Eq. (42))
    # use the GP on regular grid (q,p) for a first guess for P
    pgss = guessP([x], [y], hypp, xtrainp, ztrainp, Kyinvp, Ntest)
    res, r = newton(Pnewton, pgss, full_output=True, maxiter=205000, disp=True,
        args = (np.array([x]), np.array ([y]), l, xtrain, Kyinv, ztrain))
    return res

def calcP_expl(x,y, l, xtrain, ztrain, Kyinv):
    Kstar = np.empty((len(xtrain), 2))
    build_K(xtrain, np.hstack((x,y)), l, Kstar)
    pGP = Kstar.T.dot(Kyinv.dot(ztrain))
    res = -pGP[0] + y
    return res


    # Application of symplectic map
    #init
    pmap = np.zeros([nm, Ntest])
    pdiff = np.zeros([nm, Ntest])
    qmap = np.zeros([nm, Ntest])
    #set initial conditions
    pmap[0,:] = P0map
    qmap[0,:] = Q0map
    pdiff[0,:] = P0map
    # loop through all test points and all time steps
    for i in range(0,nm-1):
        for k in range(0, Ntest):
            # set new P including Newton for implicit Eq. (42)
            # print('nm = ', i, 'N_i = ',k)
            # print('pmap = ', pmap[i, k])
            pmap[i+1, k] = calcP(qmap[i,k], pmap[i, k], l, hypp, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv, Ntest)
            # if np.mod(pmap[i+1, k], 2*np.pi) > 0.0:
            pdiff[i+1, k] = pdiff[i, k] + (pmap[i+1, k] - pmap[i, k])
            # pmap[i+1, k] = np.mod(pmap[i+1, k], 2*np.pi) # only for standard map
            # print('nm = ', i, 'N_i = ', k, 'pdiff =', pdiff[i+1,k], 'pdiff -1 = ', pdiff[i, k], 'dp =', (pmap[i+1, k] - pmap[i, k]))
            # else:
                # pdiff[i+1, k] = pdiff[i,k] + (pmap[i+1, k] - pmap[i, k])#
                # pmap[i+1, k] = np.mod(pmap[i+1, k], 2*np.pi)
                # print('nm = ', i, 'N_i = ', k, 'pdiff =', pdiff[i+1,k], 'pdiff -1 = ', pdiff[i, k], 'dp =', (pmap[i+1, k] - pmap[i, k]))

        for k in range(0, Ntest):
            if np.isnan(pmap[i+1, k]):
                qmap[i+1,k] = np.nan
            else:
                # then: set new Q via calculating \Delta q and adding q (Eq. (43))
                dqmap = calcQ(qmap[i,k], pmap[i+1,k], xtrain, l, Kyinv, ztrain)
                # print(dqmap)
                # qmap[i+1, k] = np.mod(dqmap + qmap[i, k], 2.0*np.pi)
                # qmap[i+1, k] = np.mod(dqmap + qmap[i, k], 1)
                qmap[i+1, k] = dqmap + qmap[i, k]
    return qmap, pmap, pdiff
def applymap(nm, Ntest, l, hypp, Q0map, P0map, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv):
    # Application of symplectic map
    #init
    pmap = np.zeros([nm, Ntest])
    pdiff = np.zeros([nm, Ntest])
    qmap = np.zeros([nm, Ntest])
    #set initial conditions
    pmap[0,:] = P0map
    qmap[0,:] = Q0map
    pdiff[0,:] = P0map
    # loop through all test points and all time steps
    for i in range(0,nm-1):
        for k in range(0, Ntest):
            # set new P including Newton for implicit Eq. (42)
            # print('nm = ', i, 'N_i = ',k)
            # print('pmap = ', pmap[i, k])
            pmap[i+1, k] = calcP(qmap[i,k], pmap[i, k], l, hypp, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv, Ntest)
            # if np.mod(pmap[i+1, k], 2*np.pi) > 0.0:
            pdiff[i+1, k] = pdiff[i, k] + (pmap[i+1, k] - pmap[i, k])
            pmap[i+1, k] = np.mod(pmap[i+1, k], 2*np.pi) # only for standard map
            # print('nm = ', i, 'N_i = ', k, 'pdiff =', pdiff[i+1,k], 'pdiff -1 = ', pdiff[i, k], 'dp =', (pmap[i+1, k] - pmap[i, k]))
            # else:
                # pdiff[i+1, k] = pdiff[i,k] + (pmap[i+1, k] - pmap[i, k])#
                # pmap[i+1, k] = np.mod(pmap[i+1, k], 2*np.pi)
                # print('nm = ', i, 'N_i = ', k, 'pdiff =', pdiff[i+1,k], 'pdiff -1 = ', pdiff[i, k], 'dp =', (pmap[i+1, k] - pmap[i, k]))

        for k in range(0, Ntest):
            if np.isnan(pmap[i+1, k]):
                qmap[i+1,k] = np.nan
            else:
                # then: set new Q via calculating \Delta q and adding q (Eq. (43))
                dqmap = calcQ(qmap[i,k], pmap[i+1,k], xtrain, l, Kyinv, ztrain)
                # print(dqmap)
                qmap[i+1, k] = np.mod(dqmap + qmap[i, k], 2.0*np.pi)
                # qmap[i+1, k] = np.mod(dqmap + qmap[i, k], 1)
                # qmap[i+1, k] = dqmap + qmap[i, k]
    return qmap, pmap, pdiff

def applymap_expl(nm, Ntest, l, Q0map, P0map, xtrain, ztrain, Kyinv):
    # Application of symplectic map
    #init
    pmap = np.zeros([nm, Ntest])
    pdiff = np.zeros([nm, Ntest])
    qmap = np.zeros([nm, Ntest])
    #set initial conditions
    pmap[0,:] = P0map
    qmap[0,:] = Q0map
    pdiff[0,:] = P0map

    # loop through all test points and all time steps
    for i in range(0,nm-1):
        for k in range(0, Ntest):
            # set new P including Newton for implicit Eq (42)
            # print('nm = ', i, 'N_i = ',k)
            pmap[i+1, k] = calcP_expl(qmap[i,k], pmap[i, k], l, xtrain, ztrain, Kyinv)
            pdiff[i+1, k] = pdiff[i, k] + (pmap[i+1, k] - pmap[i, k])
            pmap[i+1, k] = np.mod(pmap[i+1, k], 2*np.pi)
            # pmap[i+1, k] = np.mod(pmap[i+1, k], 2*np.pi)
        for k in range(0, Ntest):
            if np.isnan(pmap[i+1, k]):
                qmap[i+1,k] = np.nan
            else:
                # then: set new Q via calculating \Delta q and adding q (Eq. (43))
                dqmap = calcQ(qmap[i,k], pmap[i+1,k], xtrain, l, Kyinv, ztrain)
                # print(dqmap)
                # qmap[i+1, k] = np.mod(dqmap + qmap[i, k], 2.0*np.pi)
                qmap[i+1, k] = dqmap + qmap[i, k]
    return qmap, pmap, pdiff

