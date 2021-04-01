# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:42:52 2020

@author: Katharina Rath
"""

import numpy as np
import random
import ghalton
from scipy.optimize import minimize
from func import (integrate_pendulum, build_K, buildKreg, applymap, nll_chol, energy, quality)
import tkinter
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
import scipy

#%% init parameters
Nm = 200 #mapping time (Nm = 200 for Fig. 3; Nm = 900 for Fig. 5)
N = 70 #training data (N = 20 for Fig. 3; N = 30 for Fig. 5)
U0 = 1 
nm = 200 # how often the map should be applied
dtsymp = 0.001 #integration step for producing training data
sig2_n = 1e-8 #noise**2 in observations
Ntest = 15 # number of testpoints

#define boundaries for training data
qmin = 0.0
qmax = 2*np.pi
pmin = -3.0
pmax = 3.0

#define boundaries for test data
qminmap = np.array(np.pi - 2.8)
qmaxmap = np.array(np.pi + 1.5)
pminmap = np.array(-2.3)
pmaxmap = np.array(1.8)

#set initial conditions (q0, p0) randomly for test data
random.seed(1)
q0map = np.linspace(qminmap,qmaxmap,Ntest)
p0map = np.linspace(pminmap,pmaxmap,Ntest)
q0map = random.sample(list(q0map), Ntest)
p0map = random.sample(list(p0map), Ntest)
Q0map = np.array(q0map)
P0map = np.array(p0map)

#%% calculate training data

# specify initial conditions (q0, p0), sampled from Halton sequence
seq = ghalton.Halton(2)
samples = seq.get(N)*np.array([qmax-qmin, pmax-pmin]) + np.array([qmin, pmin])

q = samples[:,0]
p = samples[:,1]

t0 = 0.0                    # starting time
t1 = dtsymp*Nm              # end time
t = np.linspace(t0,t1,Nm)   # integration points in time

#integrate pendulum to provide data
ysint = integrate_pendulum(q, p, t)

#set training data
# xtrain = (q, P), ztrain = grad F = (p-P, Q-q)
P = np.empty((N))
Q = np.empty((N))
for ik in range(0, N):    
    P[ik] = ysint[ik][1,-1]
    Q[ik] = ysint[ik][0,-1]
    
zqtrain = Q - q
zptrain = p - P

xtrain = np.hstack((q, P))
ztrain = np.concatenate((zptrain, zqtrain))


#%% set up GP
# as indicated in Algorithm 1: Semi-implicit symplectic GP map
#hyperparameter optimization of length scales (lq, lp)
log10l0 = np.array((0, 0, -1), dtype = float)

#  Step 1: Usual GP regression of P over (q,p)
#fit GP + hyperparameter optimization to have a first guess for newton for P
xtrainp = np.hstack((q, p))
ztrainp = P

sigp = 2*np.amax(np.abs(ztrainp))**2

def nll_transform2(log10hyp, sig, sig2n, x, y, N):
    hyp = 10**log10hyp
    builder = lambda x, x0, hyp, K: buildKreg(x, x0, hyp, K, N)
    return nll_chol(np.hstack((hyp, sig, [sig2n])), x, y, N, buildK=builder)
res = minimize(nll_transform2, np.array((log10l0)), args = (sigp, 1e-6, xtrainp, ztrainp.T.flatten(), N), method='L-BFGS-B')#, bounds = ((-10, 1), (-10, 1)))


lp = 10**res.x
hypp = np.hstack((lp, sigp))
print('Optimized lengthscales for regular GP: lq =', "{:.2f}".format(lp[0]), 'lp = ', "{:.2f}".format(lp[1]))
print(0.5/lp[2])
# build K and its inverse
Kp = np.zeros((N, N))
buildKreg(xtrainp, xtrainp, hypp, Kp)
Kyinvp = scipy.linalg.inv(Kp + sig2_n*np.eye(Kp.shape[0]))

# Step 2: symplectic GP regression of -Delta p and Delta q over mixed variables (q,P) according to Eq. 41
# hyperparameter optimization for lengthscales (lq, lp) and GP fitting
sig = 2*np.amax(np.abs(ztrain))**2
log10l0 = np.array((-1, 0, -0.5), dtype = float)

def nll_transform(log10hyp, sig, sig2n, x, y, N):
    hyp = 10**log10hyp
    builder = lambda x, x0, hyp, K: build_K(x, x0, hyp, K)
    return nll_chol(np.hstack((hyp, sig, [sig2n])), x, y, N, buildK = builder)

#log 10 -> BFGS
res = minimize(nll_transform, np.array((log10l0)), args = (sig, sig2_n, xtrain, ztrain.T.flatten(), 2*N), method='L-BFGS-B')#, bounds = ((-10, 1), (-10, 1)))
sol1 = 10**res.x
l = [np.abs(sol1[0]), np.abs(sol1[1]), np.abs(sol1[2])]
print('Optimized lengthscales for mixed GP: lq =', "{:.2f}".format(l[0]), 'lp = ', "{:.2f}".format(l[1]))
print('Periodicity: ', 0.5/sol1[2])
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
qmap, pmap = applymap(
    nm, Ntest, hyp, hypp, Q0map, P0map, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv)

#calculate energy
H = energy([qmap, pmap], U0)

#%% Quality of map
t0 = 0.0  # starting time
t1 = dtsymp*Nm*(nm+20) # end time
t = np.linspace(t0,t1,Nm*(nm+20)) # integration points in time
    
#integrate test data 
yinttest = np.array(integrate_pendulum(qmap[0,:], pmap[0,:], t)).T
yinttest[:,0,:] = np.mod(yinttest[:,0,:], 2*np.pi)
#%% plot results

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

plt.figure(figsize = [10,3])
plt.subplot(1,3,1)
for i in range(0, Ntest):
    plt.plot(qmap[:,i], pmap[:,i], 'k^', markersize = 0.5)
plt.ylim([-2.8, 2.8])
plt.xlabel('$q$', fontsize = 20)
plt.ylabel('$p$', fontsize = 20)
plt.tight_layout()

plt.subplot(1,3,2)
for i in range(0, Ntest):
    plt.plot(yinttest[:,0,i], yinttest[:,1,i], color = 'darkgrey', marker = 'o', linestyle = 'None', markersize = 0.5)
plt.ylim([-2.8, 2.8])    
plt.xlabel('$q$', fontsize = 20)
plt.ylabel('$p$', fontsize = 20)
plt.tight_layout()

plt.subplot(1,3,3)
for i in range(0, Ntest):
    plt.plot(yinttest[:,0,i], yinttest[:,1,i], color = 'darkgrey', marker = 'o', linestyle = 'None', markersize = 0.5)
    plt.plot(qmap[:,i], pmap[:,i], 'k^', markersize = 0.5)
plt.ylim([-2.8, 2.8])    
plt.xlabel('$q$', fontsize = 20)
plt.ylabel('$p$', fontsize = 20)
plt.tight_layout()
#%%
plt.figure()
for i in range(0, qmap.shape[1]):
    plt.plot(np.linspace(0, nm*dtsymp*Nm, nm), H[:,i]/np.mean(H[:,i]))
plt.xlabel('t')
# plt.ylim([0.5, 1.5])
plt.ylabel(r'H/$\bar{H}$')
plt.title('energyGP')

#%%
#calculate quality criteria as indicated in Eq. (40) and (41)
Eosc, gd, stdgd = quality(qmap, pmap, H, yinttest, Ntest, Nm)

print('Geometric distance: ', "{:.1e}".format(np.mean(gd)), u"\u00B1", "{:.1e}".format(stdgd))
print('Energy oscillation: ', "{:.1e}".format(np.mean(Eosc))) 

# plt.show(block=True)