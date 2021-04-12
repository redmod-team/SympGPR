# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:42:52 2020

@author: Katharina Rath
"""

import numpy as np
import random
import ghalton
from scipy.optimize import minimize
from func import (build_K, buildKreg, applymap, nll_chol, quality)
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from scipy.integrate import solve_ivp
import scipy
import time


def energy(x, U0): 
    return x[1]**2/2 + U0*(1 - np.cos(x[0]+np.pi))

def dydt_ivp(t, y):
    ydot = np.zeros([2])
    ydot[0] = y[1] 
    ydot[1] = -np.sin(y[0]+np.pi)
    return ydot

def integrate_pendulum(q0, p0, t):   
    ysint = []
    for ik in range(len(q0)):
        res_int = solve_ivp(dydt_ivp, [t[0], t[-1]], np.array((q0[ik], p0[ik])),t_eval = t,  method='LSODA', rtol = 1e-13, atol = 1e-16)
        temp = res_int.y
        ysint.append(temp)
    return ysint

def intode(y, t, dt):
    ys = np.zeros([2, len(t)])
    ys[0,0] = y[0]
    ys[1,0] = y[1]            

    for kt in range(1,len(t)):
            ys[1, kt]=ys[1, kt-1] - dt*(np.sin(ys[0, kt-1] + np.pi))
            ys[0, kt]=ys[0, kt-1] + dt*ys[1, kt]
    return ys.T

def symplEuler_pendulum(q0, p0, t, dt):
    ysint = np.zeros([len(t), 2, len(q0)]) # initial values for y

    for k in range(len(q0)):    
        ysint[:,:, k] = intode([q0[k], p0[k]], t, dt) 
    return ysint
#%% init parameters
Nm = 200 #mapping time (Nm = 200 for Fig. 3; Nm = 900 for Fig. 5)
N = 20 #training data (N = 20 for Fig. 3; N = 50 for Fig. 5)
U0 = 1
nm = 1000 # how often the map should be applied
dtsymp = 0.001 #integration step for producing training data
sig2_n = 1e-16 #noise**2 in observations
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

start = time.time()
#%% set up GP
# as indicated in Algorithm 1: Semi-implicit symplectic GP map
#hyperparameter optimization of length scales (lq, lp)
log10l0 = np.array((0, 0), dtype = float)

#  Step 1: Usual GP regression of P over (q,p)
#fit GP + hyperparameter optimization to have a first guess for newton for P
xtrainp = np.hstack((q, p))
ztrainp = P
sig2p_n = sig2_n
sigp = 2*np.amax(np.abs(ztrainp))**2

def nll_transform2(log10hyp, sig, sig2n, x, y, N):
    hyp = 10**log10hyp
    return nll_chol(np.hstack((hyp, sig, [sig2n])), x, y, N)
res = minimize(nll_transform2, np.array((log10l0)), args = (sigp, sig2p_n, xtrainp, ztrainp.T.flatten(), N), method='L-BFGS-B')#, bounds = ((-10, 1), (-10, 1)))

lp = 10**res.x
hypp = np.hstack((lp, sigp))
print('Optimized lengthscales for regular GP: lq =', "{:.2f}".format(lp[0]), 'lp = ', "{:.2f}".format(lp[1]))
# build K and its inverse
Kp = np.zeros((N, N), order='F')
buildKreg(xtrainp, xtrainp, hypp, Kp)
Kyinvp = scipy.linalg.inv(Kp + sig2_n*np.eye(Kp.shape[0]))

# Step 2: symplectic GP regression of -Delta p and Delta q over mixed variables (q,P) according to Eq. 41
# hyperparameter optimization for lengthscales (lq, lp) and GP fitting
sig = 2*np.amax(np.abs(ztrain))**2
log10l0 = np.array((-1, -1), dtype = float)
def nll_transform(log10hyp, sig, sig2n, x, y, N):
    hyp = 10**log10hyp
    return nll_chol(np.hstack((hyp, sig, [sig2n])), x, y, N)

#log 10 -> BFGS
res = minimize(nll_transform, np.array((log10l0)), args = (sig, sig2_n, xtrain, ztrain.T.flatten(), 2*N), method='L-BFGS-B', bounds = ((-10, 1), (-10, 1)))
sol1 = 10**res.x
l = [np.abs(sol1[0]), np.abs(sol1[1])]
print('Optimized lengthscales for mixed GP: lq =', "{:.2f}".format(l[0]), 'lp = ', "{:.2f}".format(l[1]))
#%%
#build K(x,x') and regularized inverse with sig2_n
# K(x,x') corresponds to L(q,P,q',P') given in Eq. (38) 
hyp = np.hstack((l, sig))
K = np.empty((2*N, 2*N), order='F')
build_K(xtrain, xtrain, hyp, K)
Kyinv = scipy.linalg.inv(K + sig2_n*np.eye(K.shape[0]))
end = time.time()
print('training time: ', "{:.2f}".format(end-start), 's')
# caluate training error
Eftrain = K.dot(Kyinv.dot(ztrain))
outtrain = mean_squared_error(ztrain, Eftrain)
print('training error', "{0:.1e}".format(outtrain))

#%% Application of symplectic map
start = time.time()
qmap, pmap = applymap(
    nm, Ntest, hyp, hypp, Q0map, P0map, xtrainp, ztrainp, Kyinvp, xtrain, ztrain, Kyinv)
end = time.time()
#%%
print("Elapsed time to predict {0:3d} steps for {1:3d} orbits: {2:.2f} s".format(nm, Ntest, end-start))
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
#%%
plt.figure()
plt.semilogy(np.linspace(0, nm, nm), np.abs(H[:,0]-H[0,0])/H[0,0])
plt.xlabel('n')
plt.ylabel(r'$Log_{10} |(H(t)-H(0))/H(0)|$', fontsize = 15)
plt.title('Implicit SympGPR')


#%% use symplectic Euler with approx. same accuracy
mf = 50
tSE = np.linspace(t0, nm*dtsymp*Nm, nm*mf)
start = time.time()
out = symplEuler_pendulum(Q0map.flatten(), P0map.flatten(), tSE, dtsymp*Nm/mf)
end = time.time()
print('Application time symplectic Euer: ', end-start)
qmapSE = np.mod(out[:,0], 2*np.pi)
pmapSE = out[:,1]
HSE = energy([qmapSE, pmapSE], U0)
EoscSE, gdSE, stdgdSE = quality(qmapSE, pmapSE, HSE, yinttest, Ntest, int(Nm/mf))
#%%
#calculate quality criteria as indicated in Eq. (40) and (41)
Eosc, gd, stdgd = quality(qmap, pmap, H, yinttest, Ntest, Nm)

print('Geometric distance: ', "{:.1e}".format(np.mean(gd)), u"\u00B1", "{:.1e}".format(stdgd))
print('Energy oscillation: ', "{:.1e}".format(np.mean(Eosc))) 


print('Sympl. Euler Geometric distance: ', "{:.1e}".format(np.mean(gdSE)), u"\u00B1", "{:.1e}".format(stdgdSE))
print('Sympl. Euler Energy oscillation: ', "{:.1e}".format(np.mean(EoscSE))) 
plt.show(block=True)
