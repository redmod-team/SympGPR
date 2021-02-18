"""
Created:  2018-08-08
Modified: 2019-03-07
Author:   Christopher Albert <albert@alumni.tugraz.at>
"""
from numpy import zeros, array, arange, append, hstack, pi, cos, sin
from scipy.optimize import fsolve, root
from common import f, qe, c
import common
from plotting import plot_orbit
from matplotlib.pyplot import *

nturn = 50      # Number of full turns
nph = 10        # Number of steps per turn
dph = 2*pi/nph  # Integrator step in phi
epspert = .005  # Absolute perturbation
m = -3          # Poloidal perturbation harmonic
n = 2           # Toroidal perturbation harmonic

r0 = 0.3
th0 = 0.0
ph0 = 0.0
f.evaluate(r0, th0, ph0)
pth0 = qe/c*f.Ath

z = zeros([3, nph*nturn + 1])
z[:,0] = [pth0, th0, 0.0]

def compute_r(pth, th, ph):
    # Compute r implicitly for given th, p_th and ph
    def rootfun(x):
        f.evaluate(x, th, ph)
        return pth - qe/c*f.Ath  # pth - pth(r, th, ph)

    sol = fsolve(rootfun, r0)
    return sol

def F_tstep(znew, zold):
    z = zeros(3)
    z[0:2] = 0.5*(zold[0:2] + znew)
    z[2] = zold[2] + 0.5*dph
    r = compute_r(z[0], z[1], z[2])
    f.evaluate(r, z[1], z[2])
    f.dAph = f.dAph - epspert*sin(m*z[1] + n*z[2]) # Apply perturbation on Aph
    ret = zeros(2)
    ret[0] = zold[0] - znew[0] + dph*(f.dAph[1] - f.dAph[0]*f.dAth[1]/f.dAth[0])  # dAph/dth
    ret[1] = zold[1] - znew[1] - dph*f.dAph[0]/f.dAth[0]                          # dAph/dpth
    return ret

from time import time
tic = time()

for kph in arange(nph*nturn):
    zold = z[:,kph]
    sol = root(F_tstep, zold[0:2], method='hybr', tol=1e-12, args=(zold))
    z[0:2, kph+1] = sol.x
    z[2, kph+1] = z[2, kph] + dph

print(f'Time taken: {time()-tic}')
print(f'Safety factor: {(z[2, nph] - z[2, 0])/(z[1, nph] - z[1, 0])}')

#%%
#plot_orbit(z)
th = z[1,:] #::nph]
r = array([compute_r(zk[0], zk[1], 0.0) for zk in z.T]).flatten()
plot(r*cos(th), r*sin(th), '.')
# %%
