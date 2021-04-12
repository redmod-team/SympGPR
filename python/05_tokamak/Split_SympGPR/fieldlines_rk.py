#%%
"""
Created:  2018-08-08
Modified: 2019-03-07
Author:   Christopher Albert <albert@alumni.tugraz.at>
"""
from numpy import zeros, array, arange, append, hstack, pi, cos, sin, mod, pi
from scipy.integrate import odeint
from scipy.optimize import fsolve
from common import f, qe, c
import common
from plotting import plot_orbit
from matplotlib.pyplot import *

nturn = 50
nph = 10
dph = 2*pi/nph
eps = .0001       # Absolute perturbation
m = -3            # Poloidal perturbation harmonic
n = 2             # Toroidal perturbation harmonic
p = array([0.0, pi/5.0])  # Perturbation phase

r0 = 0.3
th0 = 0.0
ph0 = 0.0
f.evaluate(r0, th0, ph0)
pth0 = qe/c*f.Ath

z = zeros([3, nturn*nph+1])
z[:,0] = [pth0, th0, 0.0]
rprev = r0
#%%
def compute_r(pth, th, ph):
    # Compute r implicitly for given th, p_th and ph
    global rprev
    def rootfun(x):
        f.evaluate(x, th, ph)
        return pth - qe/c*f.Ath  # pth - pth(r, th, ph)

    sol = fsolve(rootfun, rprev)
    reprev = sol
    return sol

def zdot(z, ph):
    r = compute_r(z[0], z[1], ph)
    f.evaluate(r, z[1], ph)
    f.dAph[1] = f.dAph[1] - eps*sin(m*z[1] + n*ph + p) # Apply perturbation on Aph
    ret = zeros(2)
    ret[0] = f.dAph[1] - f.dAph[0]*f.dAth[1]/f.dAth[0]   # pthdot = dAph/dth
    ret[1] = -f.dAph[0]/f.dAth[0]                        # thdot = dAph/dpth
    return ret

from time import time
tic = time()

for kturn in arange(nph*nturn):

    znew = odeint(zdot, z[0:2, kturn], [z[2, kturn], z[2, kturn]+dph])

    z[0:2, kturn+1] = znew[-1, :]
    z[2, kturn+1] = z[2, kturn] + dph

print('Time taken: {}'.format(time()-tic))
print(f'Safety factor: {(z[2, nph] - z[2, 0])/(z[1, nph] - z[1, 0])}')
#%%
figure()
plot(mod(z[1,::nph], 2*pi), z[0,::nph], ',')
#%%
figure()
plot(mod(z[1,:], 2*pi), z[0,:], ',')
#%%
#plot_orbit(z)
th = z[1,::nph]
r = array([compute_r(zk[0], zk[1], 0.0) for zk in z[:,::nph].T]).flatten()
figure()
plot(r*cos(th), r*sin(th), '.')
# %%
