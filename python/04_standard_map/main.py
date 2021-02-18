# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:42:52 2020

@author: Katharina Rath
"""

import numpy as np
from scipy.optimize import minimize
from func import (build_K, buildKreg, applymap, nll_chol, applymap_expl, nll_expl)
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
import scipy
import ghalton
import random
#%% init parameters
k = 1.0     # stochasticity parameter
N = 20      #training data
nm = 1000   # map applications
Ntest = 18  # test data
sig2_n = 1e-8 #noise**2 in observations

def StandardMap(x, k):
    outJ = (x[1] + k*np.sin(x[0]))
    outth = (x[0] + outJ)
    return [outth, outJ]


def StandardMapIterate(k, nm, N, X0):
    f = np.zeros((2, N, nm))
    f[:,:, 0] = X0
    for i in range(0, N):
        for l in range(0, nm-1):
            temp = StandardMap(f[:, i, l], k)
            f[:, i, l+1] = temp
    return f

# sample initial data points from halton sequence
seq = ghalton.Halton(2)
X0 = seq.get(N)*np.array([2*np.pi, 2*np.pi])
fmap = StandardMapIterate(k, 2, N, X0.T)

#set training data
q = fmap[0, :, 0]
p = fmap[1, :, 0]
Q = fmap[0, :, 1]
P = fmap[1, :, 1]

   
zqtrain = Q - q
zptrain = p - P

xtrain = q.flatten()
ytrain = P.flatten()
xtrain = np.hstack((q, P)).T
ztrain = np.concatenate((zptrain.flatten(), zqtrain.flatten()))

#define boundaries for test data
qminmap = np.array(0)
qmaxmap = np.array(2*np.pi)
pminmap = np.array(0)
pmaxmap = np.array(2*np.pi)

#set initial conditions (q0, p0) randomly for test data
random.seed(2)
q0map = np.linspace(qminmap,qmaxmap,Ntest)
p0map = np.linspace(pminmap,pmaxmap,Ntest)
q0map = random.sample(list(q0map), Ntest)
p0map = random.sample(list(p0map), Ntest)
Q0map = np.array(q0map)
P0map = np.array(p0map)
X0test = np.stack((Q0map, P0map)).T

# calculate reference data (test)
yinttest = StandardMapIterate(k, nm, Ntest, X0test.T)

#%% set up GP
# as indicated in Algorithm 1: Semi-implicit symplectic GP map
#hyperparameter optimization of length scales (lq, lp)
method = 'implicit'
if method == 'implicit':
    log10l0 = np.array((-1, -1), dtype = float)
    
    #  Step 1: Usual GP regression of P over (q,p)
    #fit GP + hyperparameter optimization to have a first guess for newton for P
    xtrainp = np.hstack((q, p))
    ztrainp = P-p
    
    sigp = 2*np.amax(np.abs(ztrainp))**2
    
    def nll_transform2(log10hyp, sig, sig2n, x, y, N):
        hyp = 10**log10hyp
        return nll_chol(np.hstack((hyp, sig, [sig2n])), x, y, N)
    res = minimize(nll_transform2, np.array((log10l0)), args = (sigp, sig2_n, xtrainp, ztrainp.T.flatten(), N), method='L-BFGS-B', bounds = ((-10, 1), (-10, 1)))
    
    lp = 10**res.x
    hypp = np.hstack((lp, sigp))
    print('Optimized lengthscales for regular GP: lq =', "{:.2f}".format(lp[0]), 'lp = ', "{:.2f}".format(lp[1]))
    # build K and its inverse
    Kp = np.zeros((N, N))
    buildKreg(xtrainp, xtrainp, hypp, Kp)
    Kyinvp = scipy.linalg.inv(Kp + sig2_n*np.eye(Kp.shape[0]))
    #%%
    # Step 2: symplectic GP regression of -Delta p and Delta q over mixed variables (q,P) according to Eq. 41
    # hyperparameter optimization for lengthscales (lq, lp) and GP fitting
    sig = 2*np.amax(np.abs(ztrain))**2
    log10l0 = np.array((0,0), dtype = float)
    def nll_transform(log10hyp, sig, sig2n, x, y, N):
        hyp = 10**log10hyp
        out = nll_chol(np.hstack((hyp, sig, [sig2n])), x, y, N)
        return out
    
    res = minimize(nll_transform, np.array((0,-1)), args = (sig, sig2_n, xtrain, ztrain.T.flatten(), 2*N), method='L-BFGS-B', tol= 1e-8, bounds = ((-2, 2), (-2, 2)))#, 
       
    sol1 = 10**res.x
    
    l = [np.abs(sol1[0]), np.abs(sol1[1])]
    print('Optimized lengthscales for mixed GP: lq =', "{:.2f}".format(l[0]), 'lp = ', "{:.2f}".format(l[1]), 'sig = ', "{:.2f}".format(sig))
    
    #%%
    #build K(x,x') and regularized inverse with sig2_n
    # K(x,x') corresponds to L(q,P,q',P') given in Eq. (38) 
    hyp = np.hstack((l, sig))
    K = np.empty((2*N, 2*N))
    build_K(xtrain, xtrain, hyp, K)
    Kyinv = scipy.linalg.inv(K + sig2_n*np.eye(K.shape[0]))
    
    # caluate training error
    Eftrain = K.dot(Kyinv.dot(ztrain))
    outtrain = mean_squared_error(ztrain, Eftrain)
    print('training error', "{:.1e}".format(outtrain))
    
    #%% Application of symplectic map
    outq, outp, pdiff = applymap(
        nm, Ntest, hyp, hypp, Q0map, P0map, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv)
#%%
elif method == 'explicit':
    # use kernels_expl_per_q_sq_p.pyd for sum kernel in func.py

    # Step 2: symplectic GP regression of -Delta p and Delta q over mixed variables (q,P) according to Eq. 41
    # hyperparameter optimization for lengthscales (lq, lp) and GP fitting
    # this is the explicit method
    # lq and lp are trained separately
    sig = 2*np.amax(np.abs(ztrain))**2
    log10l0 = np.array((1), dtype = float)
    def nll_transform_expl(log10hyp, sig, sig2n, x, y, N, ind):
        hyp = 10**log10hyp
        # hyp = log10hyp
        # print(hyp)
        out = nll_expl(np.hstack((hyp, sig, [sig2n])), x, y, N, ind)
        return out#[0]#, out[1]/out[0]
    
    #log 10 -> BFGS
    res_lq = minimize(nll_transform_expl, np.array((1)), args = (sig, 1e-8, xtrain, zptrain.T.flatten(), 2*N, 0), method='L-BFGS-B')#, bounds = (-2, 2))#, 
    res_lp = minimize(nll_transform_expl, np.array((1)), args = (sig, 1e-8, xtrain, zqtrain.T.flatten(), 2*N, 1), method='L-BFGS-B')#, bounds = ((-2, 2), (-2, 2)),tol= 1e-8)#, 


    sol1 = 10**res_lq.x
    sol2 = 10**res_lp.x
    print(res_lq.success)
    print(res_lp.success)
    l = np.hstack((np.abs(sol1), np.abs(sol2)))
    print('Optimized lengthscales for mixed GP: lq =', "{:.2f}".format(l[0]), 'lp = ', "{:.2f}".format(l[1]))
    hyp = np.hstack((l, sig))
    K = np.empty((2*N, 2*N))
    build_K(xtrain, xtrain, hyp, K)
    Kyinv = scipy.linalg.inv(K + sig2_n*np.eye(K.shape[0]))
    
    # caluate training error
    Eftrain = K.dot(Kyinv.dot(ztrain))
    outtrain = mean_squared_error(ztrain, Eftrain)
    print('training error', "{:.1e}".format(outtrain))
    
    outq, outp, pdiff = applymap_expl(
    nm, Ntest, hyp, Q0map, P0map, xtrain, ztrain, Kyinv)
#%%
pmap = np.mod(outp-np.pi, 2*np.pi)
qmap = np.mod(outq, 2*np.pi)
fmap = np.stack((qmap, pmap))

X0test = np.vstack((Q0map, P0map)).T
outmap = StandardMapIterate(k, nm, Ntest, X0test.T)
fintmap = np.stack((np.mod(outmap[0], 2*np.pi), np.mod(outmap[1]-np.pi, 2*np.pi)))
#%% plot results

plt.figure(figsize = [10,3])
plt.subplot(1,3,1)
for i in range(0, Ntest):
    plt.plot(qmap[:,i], pmap[:,i], 'k^', label = 'GP', markersize = 0.5)
plt.xlabel(r"$\theta$", fontsize = 20)
plt.ylabel(r"I", fontsize = 20)
plt.tight_layout()

plt.subplot(1,3,2)
plt.plot(fintmap[0], fintmap[1],  color = 'darkgrey', marker = 'o', linestyle = 'None',  markersize = 0.5)
plt.xlabel(r"$\theta$", fontsize = 20)
plt.ylabel(r"I", fontsize = 20)
plt.tight_layout()

plt.subplot(1,3,3)
plt.plot(fintmap[0], fintmap[1],  color = 'darkgrey', marker = 'o', linestyle = 'None',  markersize = 0.5)
for i in range(0, Ntest):
    plt.plot(qmap[:,i], pmap[:,i], 'k^', label = 'GP', markersize = 0.5)
plt.xlabel(r"$\theta$", fontsize = 20)
plt.ylabel(r"I", fontsize = 20)
plt.tight_layout()
plt.show(block = True)