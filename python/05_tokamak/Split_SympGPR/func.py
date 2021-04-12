import numpy as np
from scipy.optimize import newton
from scipy.linalg import solve_triangular
import scipy
from sklearn.metrics import mean_squared_error 
from fieldlines import fieldlines
from scipy.integrate import solve_ivp
from sympgpr import sympgpr
from scipy.sparse.linalg import eigsh
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
    # set up covariance matrix with derivative observations, Eq. (38)
    N = K.shape[0]//2
    N0 = K.shape[1]//2
    x0 = x0in[0:N0]
    x = xin[0:N]
    y0 = x0in[N0:2*N0]
    y = xin[N:2*N]
    sympgpr.build_k(x, y, x0, y0, hyp, K)
            
def buildKreg(xin, x0in, hyp, K):
    # set up "usual" covariance matrix for GP on regular grid (q,p)
    N = K.shape[0]
    N0 = K.shape[1]
    x0 = x0in[0:N0]
    x = xin[0:N]
    y0 = x0in[N0:2*N0]
    y = xin[N:2*N]
    sympgpr.buildkreg(x, y, x0, y0, hyp, K)

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

# negative log-posterior
def nll_chol_reg(hyp, x, y, N):
    neig = len(x)//2
    K = np.empty((N, N), order='F')
    buildKreg(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
    try:
        L = scipy.linalg.cholesky(Ky, lower = True)
        alpha = solve_cholesky(L, y)
        ret = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))
        return ret
    except:
        print('Warning! Fallback to eig solver!')
        w, Q = eigsh(Ky, neig, tol=max(1e-6*np.abs(hyp[-1]), 1e-15))
        alpha = Q.dot(np.diag(1.0/w).dot(Q.T.dot(y)))    
    

        ret = 0.5*y.T.dot(alpha) + 0.5*(np.sum(np.log(w)) + (len(x)-neig)*np.log(np.abs(hyp[-1])))
    return ret

# negative log-posterior
def nll_chol(hyp, x, y, N):
    neig = len(x)//2
    K = np.empty((N, N), order='F')
    build_K(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
    try:
        L = scipy.linalg.cholesky(Ky, lower = True)
        alpha = solve_cholesky(L, y)
        ret = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))
        return ret
    except:
        print('Warning! Fallback to eig solver!')
        w, Q = eigsh(Ky, neig, tol=max(1e-6*np.abs(hyp[-1]), 1e-15))
        alpha = Q.dot(np.diag(1.0/w).dot(Q.T.dot(y)))    
    

        ret = 0.5*y.T.dot(alpha) + 0.5*(np.sum(np.log(w)) + (len(x)-neig)*np.log(np.abs(hyp[-1])))
    return ret

def guessP(x, y, hypp, xtrainp, ztrainp, Kyinvp):
    Ntrain = len(xtrainp)//2
    return sympgpr.guessp(
        x, y, hypp, xtrainp[0:Ntrain], xtrainp[Ntrain:], ztrainp, Kyinvp)


def calcQ(x,y, xtrain, l, Kyinv, ztrain):
    Ntrain = len(xtrain)//2
    return sympgpr.calcq(
        x, y, xtrain[:Ntrain], xtrain[Ntrain:], l, Kyinv, ztrain)

def calcP(x,y, l, hypp, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv):
    Ntrain = len(xtrain)//2
    Ntrainp = len(xtrainp)//2
    return sympgpr.calcp(x, y, l, hypp, xtrainp[:Ntrainp], xtrainp[Ntrainp:],
        ztrainp, Kyinvp, xtrain[:Ntrain], xtrain[Ntrain:], ztrain, Kyinv)

def applymap_tok(nphmap, nm, Ntest, Q0map, P0map, xtrainp, ztrainp, Kyinvp, hypp, xtrain, ztrain, Kyinv, hyp):
    # Application of symplectic map
    #init
    pmap = np.zeros([nm, Ntest])
    qmap = np.zeros([nm, Ntest])
    #set initial conditions
    pmap[0,:] = P0map
    qmap[0,:] = Q0map
    i = 0
    r_gss = 0.3
    r_cut = 0.5
    # loop through all test points and all time steps
    while i < nm-nphmap:
        for m in range(0, nphmap):
            for k in range(0, Ntest): 
                if np.isnan(pmap[i, k]):
                    pmap[i+1,k] = np.nan
                else:
                    # set new P including Newton for implicit Eq
                    pmap[i+1, k] = calcP(qmap[i,k], pmap[i, k], hyp[m, :], hypp[m, :], xtrainp[:, m], ztrainp[:, m], Kyinvp[m], xtrain[:, m], ztrain[:, m], Kyinv[m])
            
            for k in range(0, Ntest):
                if np.isnan(pmap[i+1, k]):
                    qmap[i+1,k] = np.nan
                else: 
                    # then: set new Q via calculating \Delta q and adding q 
                    dqmap = calcQ(qmap[i,k], pmap[i+1,k], xtrain[:, m], hyp[m, :], Kyinv[m], ztrain[:, m])
                    qmap[i+1, k] = np.mod(dqmap + qmap[i, k], 2*np.pi)
                    ph = (2*np.pi)/nphmap*np.mod(i+1, nphmap)
                    zk = np.array([pmap[i+1, k]*1e-2, qmap[i+1,k], ph])
                    temp = fieldlines.compute_r(zk, r_gss)
                    if temp > r_cut or pmap[i+1, k] < 0.0:
                        pmap[i+1, k] = np.nan
                        qmap[i+1, k] = np.nan
            i = i + 1
    return qmap, pmap

def quality(qmap, pmap, H, ysint, Ntest, Nm):
    #geom. distance
    yref = ysint
    yref[:, 1] = np.mod(ysint[:, 1], 2*np.pi)
    gd = np.zeros([Ntest])        
    for lk in range(0,Ntest):
        gd[lk] = mean_squared_error(([pmap[1, lk], qmap[1, lk]]), yref[Nm, 0:2, lk])
    stdgd = np.std(gd[:])
    # Energy oscillation
    Eosc = np.zeros([Ntest])
    for lk in range(0, Ntest):
        Eosc[lk] = np.std(H[lk, :])/np.mean(H[lk, :])
    return Eosc, gd, stdgd

def energy(qmap, pmap, nm):
    N = qmap.shape[1]
    H = np.zeros([N, nm])
    r_gss = 0.3
    ph = np.zeros(nm)
    for i in range(0, N):
        zk = np.vstack((pmap[:, i]*1e-2, qmap[:, i], ph))
        for k in range(0, nm):
            r = fieldlines.compute_r(zk[:, k], r_gss)
            th = qmap[k, i]
            H[i, k] = -fieldlines.aph(r, th, ph[k])
    return H