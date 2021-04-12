import numpy as np
import random
import ghalton
from scipy.optimize import minimize
from func import (build_K, buildKreg, applymap_tok, nll_chol, nll_chol_reg, quality, energy)
import tkinter
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
import scipy
from scipy.integrate import solve_ivp
import cma
from mpl_toolkits.mplot3d import Axes3D
from calc_fieldlines import (X0test, yinttest, Ntest, nphmap, N, nm, q, Q, p, P, zqtrain, zptrain, ztrain, xtrain, nph)
#%% init parameters
sig2_n = 1e-14 #noise**2 in observations
Q0map = X0test[:,1]
P0map = X0test[:,0]*1e2

opt = 'cma'
def regGP(q, p, P, N):
    # Usual GP regression of P over (q,p)
    xtrainp = np.hstack((q, p))
    ztrainp = P-p

    def nll_transform2(log10hyp, sig2n, x, y, N):
        hyp = 10**log10hyp
        return nll_chol_reg(np.hstack((hyp, [sig2n])), x, y, N)
    if opt == 'lbfgs':
        res = minimize(nll_transform2, np.array((-1, 0, 1)), args = (sig2_n, xtrainp, ztrainp.T.flatten(), N), method='L-BFGS-B')
        print(res.success)
        lp = 10**res.x[0:2]
        sigp = 10**res.x[2]
    elif opt == 'cma':
        cma_opt = cma.CMAOptions()
        cma_opt.scaling_of_variables = [1, 1, 1]
        res = cma.fmin(nll_transform2, np.array((-1, 0, 1)), -0.5, cma_opt, args = (sig2_n, xtrainp, ztrainp.T.flatten(), N), restarts = 1)
        lp = 10**res[0][0:2]
        sigp = 10**res[0][2]

    hypp = np.hstack((lp, sigp))
    print('Optimized lengthscales for regular GP: lq =', "{:.2f}".format(lp[0]), 'lp = ', "{:.2f}".format(lp[1]), 'sig = ', "{:.2f}".format(sigp))
    # build K and its inverse
    Kp = np.zeros((N, N), order = 'F')
    buildKreg(xtrainp, xtrainp, hypp, Kp)
    Kyinvp = scipy.linalg.inv(Kp + sig2_n*np.eye(Kp.shape[0]))
    return Kyinvp, hypp, xtrainp, ztrainp

def nll_transform_grad(log10hyp, sig2n, x, y, N):
    hyp = log10hyp
    out = nll_chol(np.hstack((hyp, [sig2n])), x, y, N)
    return out
    
def GP(xtrain, ztrain, N):
    # symplectic GP regression of -Delta p and Delta q over mixed variables (q,P) 
    if opt == 'lbfgs':
        res = minimize(nll_transform_grad, np.array((0.5, 1.5, 1)), args = (sig2_n, xtrain, ztrain.T.flatten(), 2*N), method='L-BFGS-B')
        print(res.success)
        sol1 = res.x[0:2]
        sig = res.x[2]
    elif opt == 'cma':
        res = cma.fmin(nll_transform_grad, np.array((0.5, 2.5, 2.0)), 0.5, args = (sig2_n, xtrain, ztrain.T.flatten(), 2*N), restarts = 0)
        sol1 = res[0][0:2]
        sig = res[0][2]
    

    l = [np.abs(sol1[0]), np.abs(sol1[1])]
    print('Optimized lengthscales for mixed GP: lq =', "{:.2f}".format(l[0]), 'lp = ', "{:.2f}".format(l[1]), 'sig = ', "{:.2f}".format(sig))

    #build K(x,x') and regularized inverse with sig2_n
    # K(x,x') corresponds to L(q,P,q',P')
    hyp = np.hstack((l, sig))
    K = np.empty((2*N, 2*N), order = 'F')
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
start = time.time()
Kyinvp = np.zeros((nphmap, N, N))
hypp = np.zeros((nphmap, 3))
xtrainp = np.zeros((2*N, nphmap))
ztrainp = np.zeros((N, nphmap))
for i in range(0, nphmap):
    
    Kyinvp1, hypp1, xtrainp1, ztrainp1 = regGP(q[:, i], p[:, i], P[:, i], N)
    Kyinvp[i, :, :] = Kyinvp1
    hypp[i, :] = hypp1
    xtrainp[:, i] = xtrainp1
    ztrainp[:, i] = ztrainp1
    

# Step 2: symplectic GP regression of -Delta p and Delta q over mixed variables (q,P) 
# hyperparameter optimization for lengthscales (lq, lp) and GP fitting
Kyinv = np.zeros((nphmap, 2*N, 2*N))
hyp = np.zeros((nphmap, 3))
for i in range(0, nphmap):
    Kyinv1, hyp1 = GP(xtrain[:, i], ztrain[:, i], N)
    Kyinv[i, : , :] = Kyinv1
    hyp[i, :] = hyp1

print('Training took ', time.time() - start, ' seconds')
#%% Application of symplectic map
start = time.time()
# Q0map = np.array([0.43])
# P0map = np.array([3.23])
# Ntest = 1
# nm = 2000*nphmap
outq, outp = applymap_tok(
    nphmap, nm, Ntest, Q0map, P0map, 
    xtrainp, ztrainp, Kyinvp, hypp, xtrain, ztrain, Kyinv, hyp)
end = time.time()
print('Application took ', end-start, ' seconds')
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
    plt.plot(qmap[::nphmap,i], 1e-2*pmap[::nphmap,i], 'k^', label = 'GP', markersize = 0.5)
plt.xlabel(r"$\vartheta$", fontsize = 20)
plt.ylabel(r"$p_\vartheta$", fontsize = 20)
plt.ylim([0.007, 0.06])
plt.tight_layout()
plt.subplot(1,3,2)
plt.plot(np.mod(fintmap[::nph,1],2*np.pi), 1e-2*fintmap[::nph,0],  color = 'dodgerblue', marker = 'o', linestyle = 'None',  markersize = 0.5)
plt.xlabel(r"$\vartheta$", fontsize = 20)
plt.ylabel(r"$p_\vartheta$", fontsize = 20)
plt.ylim([0.007, 0.06])
plt.tight_layout()

plt.subplot(1,3,3)
plt.plot(np.mod(fintmap[::nph,1],2*np.pi), 1e-2*fintmap[::nph,0],  color = 'dodgerblue', marker = 'o', linestyle = 'None',  markersize = 0.5)
for i in range(0, Ntest):
    plt.plot(qmap[::nphmap,i], 1e-2*pmap[::nphmap,i], 'k^', label = 'GP', markersize = 0.5)

plt.xlabel(r"$\vartheta$", fontsize = 20)
plt.ylabel(r"$p_\vartheta$", fontsize = 20)
plt.ylim([0.007, 0.06])
plt.tight_layout()

#%% Energy oscillation + geometrical distance
H = energy(qmap[::nphmap], pmap[::nphmap], len(qmap[::nphmap]))
H_ref = energy(fintmap[::nph, 1], fintmap[::nph, 0], len(fintmap[::nph, 1]))

Eosc, gd, stdgd = quality(qmap[::nphmap], pmap[::nphmap], H, yinttest, Ntest, nph)

print('Geometric distance: ', "{:.1e}".format(np.mean(gd)), u"\u00B1", "{:.1e}".format(stdgd))
print('Energy oscillation: ', "{:.1e}".format(np.mean(Eosc)), u"\u00B1", "{:.1e}".format(np.std(Eosc)))
