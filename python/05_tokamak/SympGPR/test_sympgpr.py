#%%
import numpy as np

from func import buildKreg, build_K, guessP
from sympgpr import sympgpr

print('==============================================')
print('Testing equivalence of func.py and sympgpr.f90')
print('==============================================')

x = np.array([1.0, 2.0, 3.0])
y = np.array([0.0, 3.0, 2.0])
x0 = np.array([1.0, 2.0])
y0 = np.array([0.0, 3.0])
N = len(x)
N0 = len(x0)
hyp = np.array([0.5, 2.0, 0.4])

K = np.empty([N, N0])
buildKreg(np.hstack((x,y)), np.hstack((x0,y0)), hyp, K)
Kfort = np.empty([N, N0], order='F')
sympgpr.buildkreg(x, y, x0, y0, hyp, Kfort)

assert(np.allclose(K, Kfort, rtol=1e-12, atol=1e-12))
print('buildKreg matches for N x N0 matrix')

N1 = 1
K = np.zeros([N1, N0])
buildKreg(np.hstack((x[0:N1],y[0:N1])), np.hstack((x0,y0)), hyp, K)
Kfort = np.zeros([N1, N0], order='F')
sympgpr.buildkreg(x[0:N1], y[0:N1], x0, y0, hyp, Kfort)

assert(np.allclose(K, Kfort, rtol=1e-12, atol=1e-12))
print('buildKreg matches for 1 x N0 matrix')

#%%
K = np.empty([2*N, 2*N0])
build_K(np.hstack((x,y)), np.hstack((x0,y0)), hyp, K)
Kfort = np.empty([2*N, 2*N0], order='F')
sympgpr.build_k(x, y, x0, y0, hyp, Kfort)

assert(np.allclose(K, Kfort, rtol=1e-12, atol=1e-12))
print('build_K matches')

# %%
hypp = np.array([0.6, 1.9, 0.3])
Kyinvp = np.array([[0.9, -0.3], [0.3, 0.9]])
ztrain = np.cos(x0 + y0)
pguess = guessP(x[0], y[0], hypp, np.hstack((x0, y0)), ztrain, Kyinvp)
Kyinvp = np.array([[0.9, -0.3], [0.3, 0.9]], order='F')
pguess_fort = sympgpr.guessp(x[0], y[0], hypp, x0, y0, ztrain, Kyinvp)

assert(np.allclose(pguess, pguess_fort, rtol=1e-12, atol=1e-12))
print('pguess matches')

# %%
