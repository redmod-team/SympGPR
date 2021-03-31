#%%
import numpy as np

from sympgpr import sympgpr

x = np.array([1.0, 2.0, 3.0])
y = np.array([0.0, 3.0, 2.0])
x0 = np.array([1.0, 2.0])
y0 = np.array([0.0, 3.0])
N = len(x)
N0 = len(x0)
hyp = np.array([0.5, 2.0, 0.4])

def run_test():
    print('==============================================')
    print('Testing equivalence of func.py and sympgpr.f90')
    print('==============================================')


    K = np.empty([N, N0], order='F')
    buildKreg(np.hstack((x,y)), np.hstack((x0,y0)), hyp, K)
    Kfort = np.empty([N, N0], order='F')
    sympgpr.buildkreg(x, y, x0, y0, hyp, Kfort)

    assert(np.allclose(K, Kfort, rtol=1e-12, atol=1e-12))
    print('buildKreg matches for N x N0 matrix')
    #%%
    N1 = 1
    K = np.zeros([N1, N0], order='F')
    buildKreg(np.hstack((x[0:N1],y[0:N1])), np.hstack((x0,y0)), hyp, K)
    Kfort = np.zeros([N1, N0], order='F')
    sympgpr.buildkreg(x[0:N1], y[0:N1], x0, y0, hyp, Kfort)

    assert(np.allclose(K, Kfort, rtol=1e-12, atol=1e-12))
    print('buildKreg matches for 1 x N0 matrix')

    #%%
    K = np.empty([2*N, 2*N0], order='F')
    build_K(np.hstack((x,y)), np.hstack((x0,y0)), hyp, K)
    Kfort = np.empty([2*N, 2*N0], order='F')
    sympgpr.build_k(x, y, x0, y0, hyp, Kfort)

    assert(np.allclose(K, Kfort, rtol=1e-12, atol=1e-12))
    print('build_K matches')

    # %%
    hypp = np.array([0.6, 1.9, 0.3])
    Kyinvp = np.array([[0.9, -0.3], [0.3, 0.9]], order='F')
    ztrain = np.cos(x0 + y0)
    pguess = guessP(x[0], y[0], hypp, np.hstack((x0, y0)), ztrain, Kyinvp)
    pguess_fort = sympgpr.guessp(x[0], y[0], hypp, x0, y0, ztrain, Kyinvp)

    assert(np.allclose(pguess, pguess_fort, rtol=1e-12, atol=1e-12))

    Kyinvq = np.reshape(np.arange(16), (4,4), order='F')
    ztrainq = np.hstack((np.cos(x0 + y0), np.sin(x0 + y0)))
    q = calcQ(x[0], y[0], np.hstack((x0, y0)), hyp, Kyinvq, ztrainq)
    q_fort = sympgpr.calcq(x[0], y[0], x0, y0, hyp, Kyinvq, ztrainq)

    assert(np.allclose(q, q_fort, rtol=1e-12, atol=1e-12))
    print('calcQ matches')

    Kyinv = np.reshape(np.arange(16), (4,4), order='F')
    ztrain = np.hstack((np.cos(x0 + y0), np.sin(x0 + y0)))
    hypp = np.array([0.6, 1.9, 0.3])
    Kyinvp = np.array([[0.9, -0.3], [0.3, 0.9]], order='F')
    ztrainp = np.cos(x0 + y0)
    p = calcP(x[0], y[0], hyp, hypp, np.hstack((x0, y0)), ztrainp, Kyinvp,
        np.hstack((x0, y0)), ztrain, Kyinv)
    p_fort = sympgpr.calcp(x[0], y[0], hyp, hypp, x0, y0, ztrainp, Kyinvp,
        x0, y0, ztrain, Kyinv)

    assert(np.allclose(p, p_fort, rtol=1e-12, atol=1e-12))
    print('calcP matches')

from func import buildKreg, build_K, guessP, calcQ, calcP
run_test()

from func_old import buildKreg, build_K, guessP, calcQ, calcP
run_test()
