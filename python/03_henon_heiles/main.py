# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:42:52 2020

@author: Katharina Rath
"""

import numpy as np
import random
import ghalton
from scipy.optimize import minimize
from func import (build_K, buildKreg, applymap_henon, nll_chol, nll_grad)
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
import scipy
import time
from mpl_toolkits.mplot3d import Axes3D
import henon
#%% init parameters
N = 55 #training data
nm = 500 #map application 
sig2_n = 1e-12 #noise**2 in observations
Ntest = 37 # number of testpoints

#%% calculate training data
E_bound = 0.01
lam = 1.0
def energy(x):
    H = 1/2*(x[0]**2+x[2]**2) + 1/2*(x[1]**2+x[3]**2) + lam*(x[0]**2*x[1] - x[1]**3/3)
    return H
def ebound(x):
    Eb = 0.5*x[1]**2+0.5*x[0]**2-lam*1/3*x[0]**3
    return Eb
def calc_qdot(x):
    return np.sqrt(2*E_bound-x[1]**2-x[0]**2+lam*2/3*x[0]**3)

seq = ghalton.Halton(2)
samples_all = seq.get(2*N)*np.array([0.3, 0.3]) + np.array([-0.15, -0.15])

samples = []
for i in range(0, len(samples_all)):
    if ebound(samples_all[i, :]) < E_bound:
        samples.append(samples_all[i,:])

print('N =', N)
samples = np.array(samples)[0:N]
qdot = calc_qdot(samples[0:N].T)
samples = np.stack((np.zeros([N]),samples[0:N,0], qdot, samples[0:N,1])).T

#%%
t0 = 0.0                    # starting time
t1 = 1000                   # end time

# Setting module options
henon.henon.lam = lam
henon.henon.tmax = t1

# Tracing orbits for training data
out = []
fout = []
tout = []
plt.figure()
for ipart in range(N):
  z0 = samples[ipart, :]
  tcut, zcut, icut = henon.integrate(z0)
  out.append([tcut[:icut], zcut[:,:icut]])
  fout.append(zcut[:,:icut])
  tout.append(tcut[:icut])
  plt.plot(out[-1][1][1,:], out[-1][1][3,:], ',')


#%% cut list to minimum number of poincare sections
lenp1 = nm
for i in range(0,N):
    lenp = fout[i].shape[1]
    if lenp < lenp1:
        lenp1 = lenp

f = np.zeros((N, 4, lenp1))
tev = np.zeros((N, lenp1))
for i in range(0,N):
    f[i,:,:] = fout[i][:, 0:lenp1]
    tev[i, :] = tout[i][0:lenp1]

#set training data
# xtrain = q, ytrain = P, ztrain = grad F = (p-P, Q-q)
# rescale data for better hyperparameter optimization
q = f[:, 1, 0]*1e2
p = f[:, 3, 0]*1e2
Q = f[:, 1, 1]*1e2
P = f[:, 3, 1]*1e2

zqtrain = Q - q
zptrain = p - P

ztrain = np.hstack((zptrain, zqtrain))
xtrain = np.hstack((q, P))

zqtrain = Q - q
zptrain = p - P

xtrain = np.hstack((q, P))
ztrain = np.concatenate((zptrain, zqtrain))

random.seed(1)
q0map = np.linspace(-0.1,0.1,Ntest)
p0map = np.linspace(-0.1,0.1,Ntest)
q0map = random.sample(list(q0map), Ntest)
random.seed(0)
p0map = random.sample(list(p0map), Ntest)
Q0map = np.array(q0map)*1e2
P0map = np.array(p0map)*1e2
test_samples = np.zeros([Ntest,2])
test_samples[:,0] = np.array(q0map)
test_samples[:,1] = np.array(p0map)

#%% set up GP
# as indicated in Algorithm 1: Semi-implicit symplectic GP map
#hyperparameter optimization of length scales (lq, lp)
start = time.time()
log10l0 = np.array((0, 0), dtype = float)

#  Step 1: Usual GP regression of P over (q,p)
#fit GP + hyperparameter optimization to have a first guess for newton for P
xtrainp = np.hstack((q, p))
ztrainp = P-p

sigp = 2*np.amax(np.abs(ztrainp))**2

def nll_transform2(log10hyp, sig, sig2n, x, y, N):
    hyp = 10**log10hyp
    builder = lambda x, x0, hyp, K: buildKreg(x, x0, hyp, K, N)
    return nll_chol(np.hstack((hyp, sig, [sig2n])), x, y, N, buildK=builder)
res = minimize(nll_transform2, np.array((log10l0)), args = (sigp, sig2_n, xtrainp, ztrainp.T.flatten(), N), method='L-BFGS-B', bounds = ((-10, 1), (-10, 1)))
print(res.success)

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
def nll_transform_grad(log10hyp, sig, sig2n, x, y, N):
    hyp = 10**log10hyp
    # hyp = log10hyp
    out = nll_grad(np.hstack((hyp, sig, [sig2n])), x, y, N)
    return out[0]

#log 10 -> BFGS
res = minimize(nll_transform_grad, np.array((-1,-1)), args = (sig, sig2_n, xtrain, ztrain.T.flatten(), 2*N), method='L-BFGS-B', tol= 1e-8, bounds = ((-2, 2), (-2, 2)))#, 
sol1 = 10**res.x
l = [np.abs(sol1[0]), np.abs(sol1[1])]
print('Optimized lengthscales for mixed GP: lq =', "{:.2f}".format(l[0]), 'lp = ', "{:.2f}".format(l[1]), 'sig = ', "{:.2f}".format(sig))

#%%
#build K(x,x') and regularized inverse with sig2_n
# K(x,x') corresponds to L(q,P,q',P') 
hyp = np.hstack((l, sig))
K = np.empty((2*N, 2*N))
build_K(xtrain, xtrain, hyp, K)
Kyinv = scipy.linalg.inv(K + sig2_n*np.eye(K.shape[0]))

# caluate training error
Eftrain = K.dot(Kyinv.dot(ztrain))
outtrain = mean_squared_error(ztrain, Eftrain)
print('training error', "{:.1e}".format(outtrain))
end = time.time()
print('training time', end-start)
#%% Application of symplectic map
qmap, pmap = applymap_henon(
    nm, Ntest, hyp, hypp, Q0map, P0map, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv)

# Tracing orbits for reference plot
henon.henon.tmax = 5000
fqdot = calc_qdot(test_samples.T)
testsamples = np.stack((np.zeros(Q0map.shape), test_samples[:,0], fqdot, test_samples[:,1])).T
foutmap = []
for ipart in range(0,Ntest):
    z0 = testsamples[ipart, :]
    tcut, zcut, icut = henon.integrate(z0)
    foutmap.append(zcut[:,:icut])
  
#%%
lenp1 = nm
for i in range(0,Ntest):
    lenp = foutmap[i].shape[1]
    if lenp < lenp1:
        lenp1 = lenp

fintmap = np.zeros((Ntest, 4, lenp1))
for i in range(0,Ntest):
    fintmap[i,:,:] = foutmap[i][:, 0:lenp1]

#%% plot results

plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)

plt.figure(figsize = [10,3])
plt.subplot(1,3,1)
for i in range(0, Ntest):
    plt.plot(1e-2*qmap[:,i], 1e-2*pmap[:,i], 'k^', label = 'GP', markersize = 0.5)
plt.xlabel(r"$q_2$", fontsize = 20)
plt.ylabel(r"$p_2$", fontsize = 20)
plt.tight_layout()
plt.subplot(1,3,2)
for i in range(0, Ntest):
    plt.plot(fintmap[i, 1,:], fintmap[i, 3,:], color = 'darkgrey', marker = 'o', linestyle = 'None',  markersize = 0.5)
plt.xlabel(r"$q_2$", fontsize = 20)
plt.ylabel(r"$p_2$", fontsize = 20)
plt.tight_layout()
plt.subplot(1,3,3)
for i in range(0, Ntest):
    plt.plot(fintmap[i, 1,:], fintmap[i, 3,:], color = 'darkgrey', marker = 'o', linestyle = 'None',  markersize = 0.5)
    plt.plot(1e-2*qmap[:,i], 1e-2*pmap[:,i], 'k^', label = 'GP', markersize = 0.5)
plt.xlabel(r"$q_2$", fontsize = 20)
plt.ylabel(r"$p_2$", fontsize = 20)
plt.tight_layout()

plt.show(block = True)