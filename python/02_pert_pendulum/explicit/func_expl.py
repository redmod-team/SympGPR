# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:47:00 2020

@author: Katharina Rath
"""

import numpy as np
from scipy.linalg import solve_triangular
import scipy
from scipy.integrate import solve_ivp


from kernels import *


def f_kern(x, y, x0, y0, l):
    return kern_num(x,y,x0,y0,l[0], l[1])

def dkdx(x, y, x0, y0, l):
    return dkdx_num(x,y,x0,y0,l[0], l[1])

def dkdy(x, y, x0, y0, l):
    return dkdy_num(x,y,x0,y0,l[0], l[1])

def dkdx0(x, y, x0, y0, l):
    return dkdx0_num(x,y,x0,y0,l[0], l[1])

def dkdy0(x, y, x0, y0, l):
    return dkdy0_num(x,y,x0,y0,l[0], l[1])

def d2kdxdx0(x, y, x0, y0, l):
    return d2kdxdx0_num(x,y,x0,y0,l[0], l[1])

def d2kdydy0(x, y, x0, y0, l):
    return d2kdydy0_num(x,y,x0,y0,l[0], l[1])

def d2kdxdy0(x, y, x0, y0, l):
    return d2kdxdy0_num(x,y,x0,y0,l[0], l[1])

def d2kdydx0(x, y, x0, y0, l):
    return d2kdxdy0(x, y, x0, y0, l)


def intode(y, t, dtsymp):
    ys = np.zeros([2, len(t)])
    ys[0,0] = y[0]
    ys[1,0] = y[1]            

    for kt in range(1,len(t)):
            ys[1, kt]=ys[1, kt-1] - dtsymp*(np.sin(ys[0, kt-1] + np.pi))
            ys[0, kt]=ys[0, kt-1] + dtsymp*ys[1, kt]
    return ys.T

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

def energy(x, U0): 
    return x[1]**2/2 + U0*(1 - np.cos(x[0]))

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
    #dK[:,:] = sig*dK
    return dK

# compute log-likelihood according to RW, p.19
def solve_cholesky(L, b):
    return solve_triangular(
        L.T, solve_triangular(L, b, lower=True, check_finite=False), 
        lower=False, check_finite=False)

# negative log-posterior
def nll_chol(hyp, x, y):
    K = np.empty((len(x), len(x)))
    build_K(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(len(x)))
    L = scipy.linalg.cholesky(Ky, lower = True)
    alpha = solve_cholesky(L, y)
    ret = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))
    return ret

def nll_grad(hyp, x, y, N):
    K = np.empty((N, N))
    build_K(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
    Kyinv = np.linalg.inv(Ky)                # invert GP matrix
    alpha = Kyinv.dot(y)
    nlp_val = 0.5*y.T.dot(alpha) + 0.5*np.linalg.slogdet(Ky)[1]
    dK = build_dK(x, x, hyp[:-1])

    nlp_grad = np.array([
        -0.5*alpha.T.dot(dK[0].dot(alpha)) + 0.5*np.trace(Kyinv.dot(dK[0])),
        -0.5*alpha.T.dot(dK[1].dot(alpha)) + 0.5*np.trace(Kyinv.dot(dK[1]))
    ])

    return nlp_val, nlp_grad
def calcQ(x,y, xtrain, l, Kyinv, ztrain, Ntest):
    Kstar = np.empty((len(xtrain), 2))
    build_K(xtrain, np.hstack(([x], [y])), l, Kstar)
    qGP = Kstar.T.dot(Kyinv.dot(ztrain))
    f = qGP[1]
    
    return f, qGP[0]

def calcP(x,y, l, xtrain, ztrain, Kyinv, Ntest):
    Kstar = np.empty((len(xtrain), 2))
    build_K(xtrain, np.hstack(([x], [y])), l, Kstar)
    pGP = Kstar.T.dot(Kyinv.dot(ztrain))
    res = -pGP[0] 
    return res

def applymap(l, Q0map, P0map, xtrain, ztrain, Kyinv, Ntest, nm):
    pmap = np.zeros([nm, Ntest])
    qmap = np.zeros([nm, Ntest])
    #set initial conditions
    pmap[0,:] = P0map
    qmap[0,:] = Q0map
    for i in range(0,nm-1):
        for k in range(0, Ntest): 
            pmap[i+1, k] = pmap[i, k] + calcP(qmap[i,k], pmap[i, k], l, xtrain, ztrain, Kyinv, Ntest)
        for k in range(0, Ntest):
            if np.isnan(pmap[i+1, k]):
                qmap[i+1,k] = np.nan
            else: 
                qmap[i+1, k], dpmap = calcQ(qmap[i,k], pmap[i+1,k],xtrain, l, Kyinv, ztrain, Ntest) 
                qmap[i+1, k] = np.mod(qmap[i+1,k] + qmap[i, k], 2.0*np.pi)
    return qmap, pmap

def dydt_ivp(t, y):
    ydot = np.zeros([2])
    ydot[0] = y[1] # dx/dt
    ydot[1] = -np.sin(y[0]+np.pi)# - eps*np.sin(y_[0] - om*(t))#- 0.5*np.pi) # dpx/dt
    return ydot

def integrate_pendulum(q0, p0, t):
    ysint = np.zeros([len(t), 2, len(q0)]) # initial values for y
    ysint = []
    for ik in range(len(q0)):
        res_int = solve_ivp(dydt_ivp, [t[0], t[-1]], np.array((q0[ik], p0[ik])), t_eval = t, method='LSODA', rtol = 1e-13, atol = 1e-16)
        temp = res_int.y
        ysint.append(temp)
    return ysint

