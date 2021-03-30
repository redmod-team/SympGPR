import numpy as np

from func import buildKreg
from sympgpr import sympgpr

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
