# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:47:00 2020

@author: Katharina Rath
"""

import numpy as np
from scipy.optimize import newton
from scipy.linalg import solve_triangular
import scipy
from sklearn.metrics import mean_squared_error 

from scipy.integrate import solve_ivp

from kernels import *

def f_kern(x, y, x0, y0, l):
    return kern_num(x,y,x0,y0,l[0], l[1], l[2])

def d2kdxdx0(x, y, x0, y0, l):
    return d2kdxdx0_num(x,y,x0,y0,l[0], l[1], l[2])

def d2kdydy0(x, y, x0, y0, l):
    return d2kdydy0_num(x,y,x0,y0,l[0], l[1], l[2])

def d2kdxdy0(x, y, x0, y0, l):
    return d2kdxdy0_num(x,y,x0,y0,l[0], l[1], l[2])

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
            
def buildKreg(xin, x0in, hyp, K):
    # set up "usual" covariance matrix for GP on regular grid (q,p)
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

# negative log-posterior
def nll_chol(hyp, x, y, N, buildK = build_K):
    K = np.empty((N, N))
    build_K(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
    L = scipy.linalg.cholesky(Ky, lower = True)
    alpha = solve_cholesky(L, y)
    ret = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))
    return ret
 
def energy(x, U0): 
    return x[1]**2/2 + U0*(1 - np.cos(x[0]))

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
    return f

def calcP(x,y, l, hypp, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv, Ntest):
    # as P is given in an implicit relation, use newton to solve for P (Eq. (42))
    # use the GP on regular grid (q,p) for a first guess for P
    pgss = guessP([x], [y], hypp, xtrainp, ztrainp, Kyinvp, Ntest)
    res, r = newton(Pnewton, pgss, full_output=True, maxiter=5, disp=False,
       args = (np.array([x]), np.array([y]), l, xtrain, Kyinv, ztrain))
    return res

def applymap(nm, Ntest, l, hypp, Q0map, P0map, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv):
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
            # set new P including Newton for implicit Eq (42)
            pmap[i+1, k] = calcP(qmap[i,k], pmap[i, k], l, hypp, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv, Ntest)
        for k in range(0, Ntest):
            if np.isnan(pmap[i+1, k]):
                qmap[i+1,k] = np.nan
            else: 
                # then: set new Q via calculating \Delta q and adding q (Eq. (43))
                qmap[i+1, k] = calcQ(qmap[i,k], pmap[i+1,k], xtrain, l, Kyinv, ztrain)
                qmap[i+1, k] = np.mod(qmap[i+1,k] + qmap[i, k], 2.0*np.pi)
    return qmap, pmap

def dydt_ivp(t, y):
    ydot = np.zeros([2])
    ydot[0] = y[1] 
    ydot[1] = -np.sin(y[0]+np.pi)
    return ydot

def integrate_pendulum(q0, p0, t):   
    ysint = []
    for ik in range(len(q0)):
        res_int = solve_ivp(dydt_ivp, [t[0], t[-1]], np.array((q0[ik], p0[ik])),t_eval = t,  method='RK45', rtol = 1e-13, atol = 1e-16)
        temp = res_int.y
        ysint.append(temp)
    return ysint


def quality(qmap, pmap, H, ysint, Ntest, Nm):
    #geom. distance
    gd = np.zeros([Ntest])        
    for lk in range(0,Ntest):
        gd[lk] = mean_squared_error(([qmap[1, lk], pmap[1, lk]]),ysint[Nm, :, lk])
    stdgd = np.std(gd[:])
    # Energy oscillation
    Eosc = np.zeros([Ntest])
    for lk in range(0, Ntest):
        Eosc[lk] = np.std(H[:,lk])/np.mean(H[:,lk])
    return Eosc, gd, stdgd


def symplEuler_pendulum(q0, p0, t, dt):
    #yint = np.zeros([len(t), 2, len(q0)]) # initial values for y
    ysint = np.zeros([len(t), 2, len(q0)]) # initial values for y

    for k in range(len(q0)):    
    #     yint[:, :, k] = spint.odeint(dydt, [q0[k], p0[k]], t)
        ysint[:,:, k] = intode([q0[k], p0[k]], t, dt)
        # ysint[:,0, k] = np.mod(ysint[:,0, k], 2*np.pi)    
    return ysint