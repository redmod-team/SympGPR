# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:47:00 2020

@author: Katharina Rath
"""

import numpy as np
from scipy.linalg import solve_triangular
import scipy
from fieldlines import fieldlines
from sympgpr import sympgpr

from kernels import *

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
    N = K.shape[0]//2
    N0 = K.shape[1]//2
    x0 = x0in[0:N0]
    x = xin[0:N]
    y0 = x0in[N0:2*N0]
    y = xin[N:2*N]
    sympgpr.build_k(x, y, x0, y0, hyp, K)

def buildKreg(xin, x0in, hyp, K):
    N = K.shape[0]
    N0 = K.shape[1]
    x0 = x0in[0:N0]
    x = xin[0:N]
    y0 = x0in[N0:2*N0]
    y = xin[N:2*N]
    sympgpr.buildkreg(x, y, x0, y0, hyp, K)

def guessP(x, y, hypp, xtrainp, ztrainp, Kyinvp):
    Ntrain = len(xtrainp)//2
    return sympgpr.guessp(
        x, y, hypp, xtrainp[0:Ntrain], xtrainp[Ntrain:], ztrainp, Kyinvp)

def build_dK(xin, x0in, hyp):
    # set up covariance matrix
    N = len(xin)//2
    N0 = len(x0in)//2
    l = hyp[:-1]
    sig = hyp[-1]
    x0 = x0in[0:N0]
    x = xin[0:N]
    y0 = x0in[N0:2*N0]
    y = xin[N:2*N]
    k11 = np.empty((N0, N))
    k12 = np.empty((N0, N))
    k21 = np.empty((N0, N))
    k22 = np.empty((N0, N))

    dK = []

    for k in range(N0):
        for lk in range(N):
              k11[k,lk] = sig*d3kdxdx0dlx_num(
                  x0[k], y0[k], x[lk], y[lk], l[0], l[1])
              k21[k,lk] = sig*d3kdxdy0dlx_num(
                  x0[k], y0[k], x[lk], y[lk], l[0], l[1])
              k12[k,lk] = sig*d3kdxdy0dlx_num(
                  x0[k], y0[k], x[lk], y[lk], l[0], l[1])
              k22[k,lk] = sig*d3kdydy0dlx_num(
                  x0[k], y0[k], x[lk], y[lk], l[0], l[1])

    dK.append(np.vstack([
        np.hstack([k11, k12]),
        np.hstack([k21, k22])
    ]))

    for k in range(N0):
        for lk in range(N):
             k11[k,lk] = sig*d3kdxdx0dly_num(
                 x0[k], y0[k], x[lk], y[lk], l[0], l[1])
             k21[k,lk] = sig*d3kdxdy0dly_num(
                 x0[k], y0[k], x[lk], y[lk], l[0], l[1])
             k12[k,lk] = sig*d3kdxdy0dly_num(
                  x0[k], y0[k], x[lk], y[lk], l[0], l[1])
             k22[k,lk] = sig*d3kdydy0dly_num(
                 x0[k], y0[k], x[lk], y[lk], l[0], l[1])

    dK.append(np.vstack([
        np.hstack([k11, k12]),
        np.hstack([k21, k22])
    ]))

    for k in range(N):
        for lk in range(N0):
            k11[k,lk] = d2kdxdx0(
                x0[lk], y0[lk], x[k], y[k], l)
            k21[k,lk] = d2kdxdy0(
                 x0[lk], y0[lk], x[k], y[k], l)
            k12[k,lk] = d2kdydx0(
                 x0[lk], y0[lk], x[k], y[k], l)
            k22[k,lk] = d2kdydy0(
                x0[lk], y0[lk], x[k], y[k], l)
    dK.append(np.vstack([
        np.hstack([k11, k12]),
        np.hstack([k21, k22])
    ]))
    #dK[:,:] = sig*dK
    return dK

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
    K = np.empty((N, N), order='F')
    buildKreg(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
    L = scipy.linalg.cholesky(Ky, lower = True)
    alpha = solve_cholesky(L, y)
    ret = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))
    return ret
# negative log-posterior
def nll_chol(hyp, x, y, N):
    K = np.empty((N, N), order='F')
    build_K(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
    L = scipy.linalg.cholesky(Ky, lower = True)
    alpha = solve_cholesky(L, y)
    ret = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))
    return ret

def nll_grad(hyp, x, y, N):
    K = np.empty((N, N), order='F')
    build_K(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
    Kyinv = np.linalg.inv(Ky)                # invert GP matrix
    L = scipy.linalg.cholesky(Ky, lower = True)
    alpha = solve_cholesky(L, y)
    nlp_val = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))
    dK = build_dK(x, x, hyp[:-1])
    alpha = Kyinv.dot(y)
    # Rasmussen (5.9)
    nlp_grad = np.array([
        -0.5*alpha.T.dot(dK[0].dot(alpha)) + 0.5*np.trace(Kyinv.dot(dK[0])),
        -0.5*alpha.T.dot(dK[1].dot(alpha)) + 0.5*np.trace(Kyinv.dot(dK[1])),
        -0.5*alpha.T.dot(dK[1].dot(alpha)) + 0.5*np.trace(Kyinv.dot(dK[2]))
    ])
    return nlp_val, nlp_grad

def calcQ(x,y, xtrain, l, Kyinv, ztrain):
    Ntrain = len(xtrain)//2
    return sympgpr.calcq(
        x, y, xtrain[:Ntrain], xtrain[Ntrain:], l, Kyinv, ztrain)

def calcP(x,y, l, hypp, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv):
    Ntrain = len(xtrain)//2
    Ntrainp = len(xtrainp)//2
    return sympgpr.calcp(x, y, l, hypp, xtrainp[:Ntrainp], xtrainp[Ntrainp:],
        ztrainp, Kyinvp, xtrain[:Ntrain], xtrain[Ntrain:], ztrain, Kyinv)


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

# To use Fortran applymap_tok (no significant speedup)
# def applymap_tok(nm, Ntest, l, hypp, Q0map, P0map, xtrainp, ztrainp, Kyinvp,
#     xtrain, ztrain, Kyinv):
#     # Application of symplectic map
#     #init
#     pmap = np.zeros([nm, Ntest, 1], order='F')
#     qmap = np.zeros([nm, Ntest, 1], order='F')

#     Ntrain = len(xtrain)//2
#     Ntrainp = len(xtrainp)//2
#     sympgpr.applymap_tok(l, hypp, Q0map, P0map,
#         xtrainp[:Ntrainp], xtrainp[Ntrainp:], ztrainp, Kyinvp,
#         xtrain[:Ntrain], xtrain[Ntrain:], ztrain, Kyinv, qmap, pmap)

#     return qmap[:,:,0], pmap[:,:,0]
