# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:45:03 2020

@author: Katharina Rath
"""
#%%
import numpy as np
from scipy.optimize import minimize
from func_expl import (integrate_pendulum, build_K, energy,
                  applymap, nll_chol)
from sklearn.metrics import mean_squared_error 
import ghalton
import random
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#%% init parameters
Nm = 70 #mapping time (Nm = 70 for Fig. 4)
N = 20 #training data (N = 20 for Fig. 4)
U0 = 1 
nm = 1000 # how often the map should be applied
dtsymp = 0.001 #integration step for producing training data
sig2_n = 1e-10 #noise**2 in observations
Ntest = 15 # number of testpoints

#define boundaries for training data
qmin = 0.0
qmax = 2*np.pi
pmin = -3
pmax = 3

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

#%%
# set up GP
# as indicated in Algorithm 2: explicit symplectic GP map
# Step 1: symplectic GP regression of -Delta p and Delta q over mixed variables (q,P) according to Eq. 41
# hyperparameter optimization for lengthscales (lq, lp) and GP fitting
log10l0 = np.array((-1., -1.), dtype=float)
sig = 2*np.amax(np.abs(ztrain))**2

def nll_transform(log10hyp, sig, sig2n, x, y):
    hyp = 10**log10hyp
    return nll_chol(np.hstack((hyp, sig, [sig2n])), x, y)

res = minimize(nll_transform, log10l0, args = (sig, sig2_n, xtrain, ztrain.T.flatten()), method='L-BFGS-B')
sol1 = 10**res.x
l = [np.abs(sol1[0]), np.abs(sol1[1])]
print('Optimized lengthscales for mixed expl. GP: lq =', "{:.2f}".format(l[0]), 'lp = ', "{:.2f}".format(l[1]))
hyp = np.hstack((l, sig))
K = np.empty((2*N, 2*N))
build_K(xtrain,xtrain, hyp, K)
Kyinv = np.linalg.inv(K + sig2_n*np.eye(K.shape[0]))

#%% training error
Eftrain = K.dot(Kyinv.dot(ztrain))
outtrain = mean_squared_error(ztrain, Eftrain)
print('training error', "{:.1e}".format(outtrain))

#%% Map application
qmap, pmap = applymap(
   hyp, Q0map, P0map, xtrain, ztrain, Kyinv, Ntest, nm)
#calculate energy
H = energy([qmap, pmap], U0)

#%% Quality of map
t0 = 0.0  # starting time
t1 = dtsymp*Nm*(nm+20) # end time
t = np.linspace(t0,t1,Nm*(nm+20)) # integration points in time
    
#integrate test data approx. one bounce period 
yinttest = np.array(integrate_pendulum(qmap[0,:], pmap[0,:], t)).T
yinttest[:,0,:] = np.mod(yinttest[:,0,:], 2*np.pi)

#%% plot results

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.figure(figsize = [10, 3])
plt.subplot(1,3,1)
for i in range(0, Ntest):
    plt.plot(qmap[:,i], pmap[:,i], 'k^', markersize = 0.5)
plt.ylim([-2.8, 2.8])
plt.xlabel('$q$', fontsize = 20)
plt.ylabel('$p$', fontsize = 20)
plt.tight_layout()

plt.subplot(1,3,2)
for i in range(0, Ntest):
    plt.plot(yinttest[:,0,i], yinttest[:,1,i], color = 'dodgerblue', marker = 'o', linestyle = 'None', markersize = 0.5)
plt.ylim([-2.8, 2.8])
plt.xlabel('$q$', fontsize = 20)
plt.ylabel('$p$', fontsize = 20)
plt.tight_layout()

plt.subplot(1,3,3)
for i in range(0, Ntest):
    plt.plot(yinttest[:,0,i], yinttest[:,1,i], color = 'dodgerblue', marker = 'o', linestyle = 'None', markersize = 0.5)
    plt.plot(qmap[:,i], pmap[:,i], 'k^', markersize = 0.5)
plt.ylim([-2.8, 2.8])
plt.xlabel('$q$', fontsize = 20)
plt.ylabel('$p$', fontsize = 20)
plt.tight_layout()

plt.figure()
plt.semilogy((H[:,0]-H[0,0])/H[0,0], 'k')
plt.xlabel('n', fontsize = 20)
plt.ylabel(r'$Log_{10} |(H(t)-H(0))/H(0)|$', fontsize = 15)
plt.title('Energy Expl. SympGPR')
plt.tight_layout()
plt.show()