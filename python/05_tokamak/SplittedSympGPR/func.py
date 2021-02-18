import numpy as np
from scipy.optimize import newton
from scipy.linalg import solve_triangular
import scipy
from sklearn.metrics import mean_squared_error 
from fieldlines import fieldlines
from scipy.integrate import solve_ivp

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
    K = np.empty((N, N))
    buildKreg(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
    L = scipy.linalg.cholesky(Ky, lower = True)
    alpha = solve_cholesky(L, y)
    ret = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))
    return ret

# negative log-posterior
def nll_chol(hyp, x, y, N, buildK = build_K):
    K = np.empty((N, N))
    build_K(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(N))
    L = scipy.linalg.cholesky(Ky, lower = True)
    alpha = solve_cholesky(L, y)
    ret = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))
    return ret

def nll_grad(hyp, x, y, N):
    K = np.empty((N, N))
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
    # as P is given in an implicit relation, use newton to solve for P
    # use the GP on regular grid (q,p) for a first guess for P
    pgss = guessP([x], [y], hypp, xtrainp, ztrainp, Kyinvp, Ntest)
    res, r = newton(Pnewton, pgss, full_output=True, maxiter=500, disp=True,
        args = (np.array([x]), np.array ([y]), l, xtrain, Kyinv, ztrain))
    return res



def applymap_tok(nm, Ntest, Q0map, P0map, 
        xtrainp1, ztrainp1, Kyinvp1, hypp1, xtrain1, ztrain1, Kyinv1, hyp1, 
        xtrainp2, ztrainp2, Kyinvp2, hypp2, xtrain2, ztrain2, Kyinv2, hyp2,
        xtrainp3, ztrainp3, Kyinvp3, hypp3, xtrain3, ztrain3, Kyinv3, hyp3,
        xtrainp4, ztrainp4, Kyinvp4, hypp4, xtrain4, ztrain4, Kyinv4, hyp4):
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
    while i < nm-4:
        for k in range(0, Ntest): 
            if np.isnan(pmap[i, k]):
                pmap[i+1,k] = np.nan
            else:
                # set new P including Newton for implicit Eq
                pmap[i+1, k] = calcP(qmap[i,k], pmap[i, k], hyp1, hypp1, xtrainp1, ztrainp1, Kyinvp1, xtrain1, ztrain1, Kyinv1, Ntest)
            
                ph = 2*np.pi/4
                zk = np.array([pmap[i+1, k]*1e-2, qmap[i,k], ph])
                temp = fieldlines.compute_r(zk, r_gss)
                if temp > r_cut or pmap[i+1, k] < 0.0:
                    pmap[i+1, k] = np.nan
        for k in range(0, Ntest):
            if np.isnan(pmap[i+1, k]):
                qmap[i+1,k] = np.nan
            else: 
                # then: set new Q via calculating \Delta q and adding q 
                dqmap = calcQ(qmap[i,k], pmap[i+1,k], xtrain1, hyp1, Kyinv1, ztrain1)
                qmap[i+1, k] = np.mod(dqmap + qmap[i, k], 2*np.pi)
        #
        i = i + 1
        for k in range(0, Ntest): 
            if np.isnan(pmap[i, k]):
                pmap[i+1,k] = np.nan
            else:
                # set new P including Newton for implicit Eq. (42)
                pmap[i+1, k] = calcP(qmap[i,k], pmap[i, k], hyp2, hypp2, xtrainp2, ztrainp2, Kyinvp2, xtrain2, ztrain2, Kyinv2, Ntest)
                ph = 2*np.pi/2
                zk = np.array([pmap[i+1, k]*1e-2, qmap[i,k], ph])
                temp = fieldlines.compute_r(zk, r_gss)
                if temp > r_cut or pmap[i+1, k] < 0.0:
                    pmap[i+1, k] = np.nan
        
        for k in range(0, Ntest):
            if np.isnan(pmap[i+1, k]):
                qmap[i+1,k] = np.nan
            else: 
                # then: set new Q via calculating \Delta q and adding q
                dqmap = calcQ(qmap[i,k], pmap[i+1,k], xtrain2, hyp2, Kyinv2, ztrain2)
                qmap[i+1, k] = np.mod(dqmap + qmap[i, k], 2*np.pi)
        
        i = i + 1
        for k in range(0, Ntest): 
            if np.isnan(pmap[i, k]):
                pmap[i+1,k] = np.nan
            else:
                # set new P including Newton for implicit Eq
                pmap[i+1, k] = calcP(qmap[i,k], pmap[i, k], hyp3, hypp3, xtrainp3, ztrainp3, Kyinvp3, xtrain3, ztrain3, Kyinv3, Ntest)
                ph = 2*3*np.pi/4
                zk = np.array([pmap[i+1, k]*1e-2, qmap[i,k], ph])
                temp = fieldlines.compute_r(zk, r_gss)
                if temp > r_cut or pmap[i+1, k] < 0.0:
                    pmap[i+1, k] = np.nan
        for k in range(0, Ntest):
            if np.isnan(pmap[i+1, k]):
                qmap[i+1,k] = np.nan
            else: 
                # then: set new Q via calculating \Delta q and adding q 
                dqmap = calcQ(qmap[i,k], pmap[i+1,k], xtrain3, hyp3, Kyinv3, ztrain3)
                qmap[i+1, k] = np.mod(dqmap + qmap[i, k], 2*np.pi)
        
        i = i + 1
        for k in range(0, Ntest): 
            if np.isnan(pmap[i, k]):
                pmap[i+1,k] = np.nan
            else:
                # set new P including Newton for implicit Eq.
                pmap[i+1, k] = calcP(qmap[i,k], pmap[i, k], hyp4, hypp4, xtrainp4, ztrainp4, Kyinvp4, xtrain4, ztrain4, Kyinv4, Ntest)
                ph = 2*np.pi
                zk = np.array([pmap[i+1, k]*1e-2, qmap[i,k], ph])
                temp = fieldlines.compute_r(zk, r_gss)
                if temp > r_cut or pmap[i+1, k] < 0.0:
                    pmap[i+1, k] = np.nan
        
        for k in range(0, Ntest):
            if np.isnan(pmap[i+1, k]):
                qmap[i+1,k] = np.nan
            else: 
                # then: set new Q via calculating \Delta q and adding q 
                dqmap = calcQ(qmap[i,k], pmap[i+1,k], xtrain4, hyp4, Kyinv4, ztrain4)
                qmap[i+1, k] = np.mod(dqmap + qmap[i, k], 2*np.pi)
        i = i + 1
    return qmap, pmap
