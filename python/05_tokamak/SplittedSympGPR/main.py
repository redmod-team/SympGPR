import numpy as np
import random
import ghalton
from scipy.optimize import minimize
from func import (build_K, buildKreg, applymap_tok, nll_chol, nll_chol_reg, nll_grad)
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
import scipy
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from calc_fieldlines import (q1, Q1, p1, P1, zqtrain1, zptrain1, xtrain1, ztrain1, X0test, yinttest, Ntest, nphmap, N, nm,
                             q2, Q2, p2, P2, zqtrain2, zptrain2, xtrain2, ztrain2,
                             q3, Q3, p3, P3, zqtrain3, zptrain3, xtrain3, ztrain3,
                             q4, Q4, p4, P4, zqtrain4, zptrain4, xtrain4, ztrain4)
#%% init parameters
sig2_n = 1e-8 #noise**2 in observations
Q0map = X0test[:,1]
P0map = X0test[:,0]*1e2
def regGP(q, p, P, N):
    # Usual GP regression of P over (q,p)
    xtrainp = np.hstack((q, p))
    ztrainp = P-p

    def nll_transform2(log10hyp, sig2n, x, y, N):
        hyp = 10**log10hyp
        return nll_chol_reg(np.hstack((hyp, [sig2n])), x, y, N)
    res = minimize(nll_transform2, np.array((-1,-1, 1)), args = (sig2_n, xtrainp, ztrainp.T.flatten(), N), method='L-BFGS-B')
    print(res.success)
    
    lp = 10**res.x[0:2]
    sigp = 10**res.x[2]
    hypp = np.hstack((lp, sigp))
    print('Optimized lengthscales for regular GP: lq =', "{:.2f}".format(lp[0]), 'lp = ', "{:.2f}".format(lp[1]), 'sig = ', "{:.2f}".format(sigp))
    # build K and its inverse
    Kp = np.zeros((N, N))
    buildKreg(xtrainp, xtrainp, hypp, Kp)
    Kyinvp = scipy.linalg.inv(Kp + sig2_n*np.eye(Kp.shape[0]))
    return Kyinvp, hypp, xtrainp, ztrainp

def nll_transform_grad(log10hyp, sig2n, x, y, N):
    hyp = log10hyp
    out = nll_grad(np.hstack((hyp, [sig2n])), x, y, N)
    return out[0]
    
def GP(xtrain, ztrain, N):
    # symplectic GP regression of -Delta p and Delta q over mixed variables (q,P) 
    res = minimize(nll_transform_grad, np.array((0.5, 0.5, 1)), args = (sig2_n, xtrain, ztrain.T.flatten(), 2*N), method='L-BFGS-B')
    sol1 = res.x[0:2]
    sig = res.x[2]
    print(res.success)
    l = [np.abs(sol1[0]), np.abs(sol1[1])]
    print('Optimized lengthscales for mixed GP: lq =', "{:.2f}".format(l[0]), 'lp = ', "{:.2f}".format(l[1]), 'sig = ', "{:.2f}".format(sig))

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
    
    return Kyinv, hyp
#%% set up GP
# as indicated in Algorithm 1: Semi-implicit symplectic GP map
#hyperparameter optimization of length scales (lq, lp)

#  Step 1: Usual GP regression of P over (q,p)
#fit GP + hyperparameter optimization to have a first guess for newton for P
Kyinvp1, hypp1, xtrainp1, ztrainp1 = regGP(q1, p1, P1, N)
Kyinvp2, hypp2, xtrainp2, ztrainp2 = regGP(q2, p2, P2, N)
Kyinvp3, hypp3, xtrainp3, ztrainp3 = regGP(q3, p3, P3, N)
Kyinvp4, hypp4, xtrainp4, ztrainp4 = regGP(q4, p4, P4, N)

#%%
# Step 2: symplectic GP regression of -Delta p and Delta q over mixed variables (q,P) 
# hyperparameter optimization for lengthscales (lq, lp) and GP fitting

Kyinv1, hyp1 = GP(xtrain1, ztrain1, N)
Kyinv2, hyp2 = GP(xtrain2, ztrain2, N)
Kyinv3, hyp3 = GP(xtrain3, ztrain3, N)
Kyinv4, hyp4 = GP(xtrain4, ztrain4, N)

#%% Application of symplectic map

outq, outp = applymap_tok(
    nm, Ntest, Q0map, P0map, 
    xtrainp1, ztrainp1, Kyinvp1, hypp1, xtrain1, ztrain1, Kyinv1, hyp1, 
    xtrainp2, ztrainp2, Kyinvp2, hypp2, xtrain2, ztrain2, Kyinv2, hyp2,
    xtrainp3, ztrainp3, Kyinvp3, hypp3, xtrain3, ztrain3, Kyinv3, hyp3,
    xtrainp4, ztrainp4, Kyinvp4, hypp4, xtrain4, ztrain4, Kyinv4, hyp4)

 
qmap = outq
pmap = outp
fmap = np.stack((qmap, pmap))

fintmap = yinttest
fintmap[:,0] = yinttest[:,0]
fintmap[:,1] = yinttest[:,1]
#%% plot results

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.figure(figsize = [10,3])
plt.subplot(1,3,1)
for i in range(0, Ntest):
    plt.plot(np.mod(qmap[::4,i], 2*np.pi), 1e-2*pmap[::4,i], 'k^', label = 'GP', markersize = 0.5)
plt.xlabel(r"$\vartheta$", fontsize = 20)
plt.ylabel(r"$p_\vartheta$", fontsize = 20)
plt.ylim([0.007, 0.06])
plt.tight_layout()
plt.subplot(1,3,2)
plt.plot(np.mod(fintmap[::32,1],2*np.pi), 1e-2*fintmap[::32,0],  color = 'darkgrey', marker = 'o', linestyle = 'None',  markersize = 0.5)
plt.xlabel(r"$\vartheta$", fontsize = 20)
plt.ylabel(r"$p_\vartheta$", fontsize = 20)
plt.ylim([0.007, 0.06])
plt.tight_layout()

plt.subplot(1,3,3)
plt.plot(np.mod(fintmap[::32,1],2*np.pi), 1e-2*fintmap[::32,0],  color = 'darkgrey', marker = 'o', linestyle = 'None',  markersize = 0.5)
for i in range(0, Ntest):
    plt.plot(np.mod(qmap[::4,i], 2*np.pi), 1e-2*pmap[::4,i], 'k^', label = 'GP', markersize = 0.5)

plt.xlabel(r"$\vartheta$", fontsize = 20)
plt.ylabel(r"$p_\vartheta$", fontsize = 20)
plt.ylim([0.007, 0.06])
plt.tight_layout()
plt.show()
