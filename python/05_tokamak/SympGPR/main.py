# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:42:52 2020

@author: Katharina Rath
"""

import numpy as np
import random
from scipy.optimize import minimize
from func import (build_K, buildKreg, applymap_tok, nll_chol_reg, nll_chol, nll_grad)
import tkinter
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy
from calc_fieldlines import q, Q, p, P, zqtrain, zptrain, ztrain, X0test, yinttest, xtrain, Ntest, nm, N
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
#%% init parameters

sig2_n = 1e-12 #noise**2 in observations

Q0map = X0test[:,1]
P0map = X0test[:,0]*1e2

#%% set up GP
# as indicated in Algorithm 1: Semi-implicit symplectic GP map
#hyperparameter optimization of length scales (lq, lp)

#  Step 1: Usual GP regression of P over (q,p)
#fit GP + hyperparameter optimization to have a first guess for newton for P
xtrainp = np.hstack((q, p))
ztrainp = P-p

def nll_transform2(log10hyp, sig2n, x, y, N):
    hyp = 10**log10hyp
    return nll_chol_reg(np.hstack((hyp, [sig2n])), x, y, N)
res = minimize(nll_transform2, np.array((0, 0, 1)), args = (1e-8, xtrainp,
    ztrainp.T.flatten(), N), method='L-BFGS-B')
print(res.success)

lp = 10**res.x[0:2]
sigp = 10**res.x[2]
hypp = np.hstack((lp, sigp))
print('Optimized lengthscales for regular GP: lq =', "{:.2f}".format(lp[0]),
    'lp = ', "{:.2f}".format(lp[1]), 'sig = ', "{:.2f}".format(sigp))
# build K and its inverse
Kp = np.zeros((N, N), order='F')
buildKreg(xtrainp, xtrainp, hypp, Kp)
Kyinvp = scipy.linalg.inv(Kp + sig2_n*np.eye(Kp.shape[0]))
#%%
# Step 2: symplectic GP regression of -Delta p and Delta q
# over mixed variables (q,P)
# hyperparameter optimization for lengthscales (lq, lp) and GP fitting

def nll_transform_grad(log10hyp, sig2n, x, y, N):
    hyp = log10hyp
    return nll_chol(np.hstack((hyp, [sig2n])), x, y, N)

res = minimize(nll_transform_grad, np.array((0.5, 0.5, 10)),
    args = (sig2_n, xtrain, ztrain.T.flatten(), 2*N), method='L-BFGS-B')

sol1 = res.x[0:2]
sig = res.x[2]
print(res.success)
l = [np.abs(sol1[0]), np.abs(sol1[1])]
print('Optimized lengthscales for mixed GP: lq =', "{:.2f}".format(l[0]),
    'lp = ', "{:.2f}".format(l[1]), 'sig = ', "{:.2f}".format(sig))

#%%
#build K(x,x') and regularized inverse with sig2_n
# K(x,x') corresponds to L(q,P,q',P') given in Eq. (38)
hyp = np.hstack((l, sig))
K = np.empty((2*N, 2*N), order='F')
build_K(xtrain, xtrain, hyp, K)
Kyinv = scipy.linalg.inv(K + sig2_n*np.eye(K.shape[0]))

# caluate training error
Eftrain = K.dot(Kyinv.dot(ztrain))
outtrain = mean_squared_error(ztrain, Eftrain)
print('training error', "{:.1e}".format(outtrain))

#%% Application of symplectic map
import pickle
data = (nm, Ntest, hyp, hypp, Q0map, P0map, xtrainp, ztrainp,
    Kyinvp, xtrain, ztrain, Kyinv)
pickle.dump(data, open('test.pickle', 'wb'))
qmap, pmap = applymap_tok(
    nm, Ntest, hyp, hypp, Q0map, P0map, xtrainp, ztrainp,
    Kyinvp, xtrain, ztrain, Kyinv)

fmap = np.stack((qmap, pmap))

#%% plot results

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.figure(figsize = [10,4])
plt.subplot(1,3,1)
for i in range(0, Ntest):
    plt.plot(np.mod(qmap[:,i], 2*np.pi), 1e-2*pmap[:,i], 'k^',
        label = 'GP', markersize = 0.5)
plt.xlabel(r"$\vartheta$", fontsize = 20)
plt.ylabel(r"$p_\vartheta$", fontsize = 20)
plt.ylim([0.007, 0.06])
plt.tight_layout()

plt.subplot(1,3,2)
plt.plot(np.mod(yinttest[::32,1],2*np.pi), yinttest[::32,0],
    color = 'darkgrey', marker = 'o', linestyle = 'None',  markersize = 0.5)
plt.xlabel(r"$\vartheta$", fontsize = 20)
plt.ylabel(r"$p_\vartheta$", fontsize = 20)
plt.ylim([0.007, 0.06])
plt.tight_layout()

plt.subplot(1,3,3)
for i in range(0, Ntest):
    plt.plot(np.mod(qmap[:,i], 2*np.pi), 1e-2*pmap[:,i], 'k^',
        label = 'GP', markersize = 0.5)
plt.plot(np.mod(yinttest[::32,1],2*np.pi), yinttest[::32,0],
    color = 'darkgrey', marker = 'o', linestyle = 'None',  markersize = 0.5)
plt.xlabel(r"$\vartheta$", fontsize = 20)
plt.ylabel(r"$p_\vartheta$", fontsize = 20)
plt.ylim([0.007, 0.06])
plt.tight_layout()
# plt.show(block = True)
