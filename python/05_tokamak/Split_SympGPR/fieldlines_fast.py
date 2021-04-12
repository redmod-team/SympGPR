#%%
"""
Created:  2018-08-08
Modified: 2019-03-07
Author:   Christopher Albert <albert@alumni.tugraz.at>
"""
from numpy import zeros, array, arange, append, hstack, pi, cos, sin, mod
from scipy.optimize import fsolve, root
import common
from plotting import plot_orbit
from matplotlib.pyplot import *
from fieldlines import fieldlines

#
# Attention: Variables are [pth, th, ph], so momentum in first component and
#            "time" in third. Change to usual Z = z[:,1::-1] before using!
#

nturn = 1000 # Number of full turns
nph = 32     # Number of steps per turn

r0 = 0.3
th0 = 0.0
ph0 = 0.0

fieldlines.init(nph=nph, am=-3, an=2, aeps=.001, aphase=0.0, arlast=r0)

pth0 = fieldlines.ath(r0, th0, ph0)

#%%

z = zeros([nph*nturn + 1, 3])
z[0,:] = [pth0, th0, 0.0]

from time import time
tic = time()

for kph in arange(nph*nturn):
    z[kph+1, :] = z[kph, :]
    fieldlines.timestep(z[kph+1, :])


print(f'Time taken: {time()-tic}')
print(f'Safety factor: {(z[nph, 2] - z[0, 2])/(z[nph, 1] - z[0, 1])}')

#%%
figure()
plot(z[::nph,0], mod(z[::nph,1], 2*pi), ',')

#%%
figure()
th = z[::nph,1]
r = array([fieldlines.compute_r(zk, 0.3) for zk in z[::nph,:]]).flatten()
plot(r*cos(th), r*sin(th), ',')
