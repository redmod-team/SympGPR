# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:42:52 2020

@author: Katharina Rath
"""

import numpy as np
from scipy.optimize import minimize
from func_expl import (build_K, applymap, nll_chol, nll_grad)
# import tkinter
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
import scipy
import time
from calc_poincare import ztrain, xtrain, q, p, P, Q, N, qs, ps, Ntest, yinttest, nm
#%% init parameters
U0 = 1 
sig2_n = 1e-8 #noise**2 in observations

# #%% set up GP
# # as indicated in Algorithm 1: Semi-implicit symplectic GP map
# #hyperparameter optimization of length scales (lq, lp)
# lp0 = np.array((0.5, 0.5), dtype = float)

# #  Step 1: Usual GP regression of P over (q,p)
# #fit GP + hyperparameter optimization to have a first guess for newton for P
# xtrainp = np.hstack((q, p)).T
# ztrainp = P

# sigp = 2*np.amax(np.abs(ztrainp))**2

# def nll_transform2(log10hyp, sig, sig2n, x, y, N):
#     hyp = log10hyp
#     return nll_grad_reg(np.hstack((hyp, sig, [sig2n])), x, y, N)
# res = minimize(nll_transform2, np.array((lp0)), args = (sigp, sig2_n, xtrainp, ztrainp.T.flatten(), N), method='L-BFGS-B', jac = True)

# lp = res.x
# hypp = np.hstack((lp, sigp))
# print('Optimized lengthscales for regular GP: lq =', "{:.2f}".format(lp[0]), 'lp = ', "{:.2f}".format(lp[1]))

# # build K and its inverse
# Kp = np.zeros((N, N))
# buildKreg(xtrainp, xtrainp, hypp, Kp)
# Kyinvp = scipy.linalg.inv(Kp + sig2_n*np.eye(Kp.shape[0]))
#%%
# Step 2: symplectic GP regression of -Delta p and Delta q over mixed variables (q,P) according to Eq. 41
# hyperparameter optimization for lengthscales (lq, lp) and GP fitting
l0 = np.array((2.0, 2.0), dtype = float)
sig = 2*np.amax(np.abs(ztrain))**2
def nll_transform_grad(log10hyp, sig, sig2n, x, y, N):
    hyp = log10hyp
    return nll_grad(np.hstack((hyp, sig, [sig2n])), x, y, N)


res = minimize(nll_transform_grad, np.array((l0)), args = (sig, sig2_n, xtrain, ztrain.T.flatten(), 2*N), method='L-BFGS-B',jac=True)
sol1 = res.x
l = [np.abs(sol1[0]), np.abs(sol1[1])]
print('Optimized lengthscales for mixed GP: lq =', "{:.2f}".format(l[0]), 'lp = ', "{:.2f}".format(l[1]))
print('Opt. Success', res.success)
print('NLL', res.fun)

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

start = time.time()
qmap, pmap = applymap(hyp, qs, ps, xtrain, ztrain, Kyinv, Ntest, nm)
    # nm, Ntest, hyp, hypp, qs, ps, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv)
end = time.time()
print('Time needed SGPR: ', end-start)

#l, Q0map, P0map, xtrain, ztrain, Kyinv, Ntest, nm
#%% plot results
plt.figure(figsize = [10,3])
plt.subplot(1,3,1)
for i in range(0, Ntest):
    plt.plot(qmap[:,i], pmap[:,i], 'k^', markersize = 0.5)
plt.xlabel('$q$', fontsize = 20)
plt.ylabel('$p$', fontsize = 20)
plt.tight_layout()

plt.subplot(1,3,2)
for i in range(0, Ntest):
    plt.plot(yinttest[0,i,:], yinttest[1,i,:], color = 'darkgrey', marker = 'o', linestyle = 'None',  markersize = 0.5)
plt.xlabel('$q$', fontsize = 20)
plt.ylabel('$p$', fontsize = 20)
plt.tight_layout()

plt.subplot(1,3,3)
for i in range(0, Ntest):
    plt.plot(yinttest[0,i,:], yinttest[1,i,:], color = 'darkgrey', marker = 'o', linestyle = 'None', markersize = 0.5)
    plt.plot(qmap[:,i], pmap[:,i], 'k^', markersize = 0.5)
plt.xlabel('$q$', fontsize = 20)
plt.ylabel('$p$', fontsize = 20)
plt.tight_layout()
# plt.show(block = True)