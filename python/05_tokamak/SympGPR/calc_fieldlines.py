from numpy import zeros, array, arange, append, hstack, pi, cos, sin, mod, linspace, concatenate,vstack
import numpy as np
from scipy.optimize import fsolve, root
from common import f, qe, c
import common
import tkinter
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from fieldlines import fieldlines
import ghalton
#from profit.util.halton import halton
from random import randint
import random
#
# Attention: Variables are [pth, th, ph], so momentum in first component and
#            "time" in third. Change to usual Z = z[:,1::-1] before using!
#
N = 78
nm = 2000
nturn = 2 # Number of full turns
nph = 32     # Number of steps per turn

nphmap = 1
eps = 0.001
seq = ghalton.Halton(3)
X0 = seq.get(N)*array([0.26, 2*pi, 0])+array([0.1, 0, 0])#
yint = zeros([nph*nturn + 1, 3, N])
q = zeros([N])
p = zeros([N])
Q = zeros([N])
P = zeros([N])


for ipart in range(0, N):
    
    r0 = X0[ipart, 0]
    th0 = X0[ipart, 1]
    ph0 = X0[ipart, 2]
    fieldlines.init(nph=nph, am=-3, an=2, aeps=eps, aphase=0.0, arlast=r0)
    pth0 = qe/c*fieldlines.ath(r0, th0, ph0)
    
    #%%
    
    z = zeros([nph*nturn + 1, 3])
    z[0,:] = [pth0, th0, 0.0]
    
    from time import time
    tic = time()
    
    for kph in arange(nph*nturn):
        z[kph+1, :] = z[kph, :]
        fieldlines.timestep(z[kph+1, :])
    yint[:,:,ipart] = z
    
#%%
    th = z[::nph,1]
    r = array([fieldlines.compute_r(zk, 0.3) for zk in z[::nph,:]]).flatten()
    
#%%

print('Integration of training data finished')

q = yint[0,1]
p = yint[0,0]*1e2
Q = yint[int(nph/nphmap),1]
P = yint[int(nph/nphmap),0]*1e2
zqtrain = Q - q
zptrain = p - P

#%%

xtrain = q.flatten()
ytrain = P.flatten()
xtrain = hstack((q, P)).T
ztrain1 = zptrain.flatten()
ztrain2 = zqtrain.flatten()
ztrain = concatenate((ztrain1, ztrain2))

# set test data and calculate reference orbits
Ntest = 30
nturntest = nm

qminmap = 0.15
qmaxmap = 0.25
pminmap = 0
pmaxmap = 2*np.pi

qminplt = 0.15
qmaxplt = 0.31

random.seed(1)
q0map = linspace(qminmap,qmaxmap,Ntest)
p0map = linspace(pminmap,pmaxmap,Ntest)
q0map = random.sample(list(q0map), Ntest)
p0map = random.sample(list(p0map), Ntest)
Q0map = np.array(q0map)
P0map = np.array(p0map)
X0test = np.stack((Q0map, P0map, np.zeros(Ntest))).T

random.seed(1)
q0map = linspace(qminplt,qmaxplt,Ntest)
p0map = linspace(pminmap,pmaxmap,Ntest)
q0map = random.sample(list(q0map), Ntest)
p0map = random.sample(list(p0map), Ntest)
Q0map = np.array(q0map)
P0map = np.array(p0map)
X0testplt = np.stack((Q0map, P0map, np.zeros(Ntest))).T

yinttest = zeros([nph*nturntest + 1, 3, Ntest])
# plt.figure()
for ipart in range(0, Ntest):
    fieldlines.init(nph=nph, am=-3, an=2, aeps=eps, aphase=0.0, arlast=X0test[ipart,0])
    X0test[ipart, 0] = qe/c*fieldlines.ath(X0test[ipart,0], X0test[ipart,1], X0test[ipart,2])
    
    fieldlines.init(nph=nph, am=-3, an=2, aeps=eps, aphase=0.0, arlast=X0testplt[ipart,0])
    X0testplt[ipart, 0] = qe/c*fieldlines.ath(X0testplt[ipart,0], X0testplt[ipart,1], X0testplt[ipart,2])
    temp = zeros([nph*nturntest + 1, 3])
    temp[0,:] = [X0testplt[ipart,0], X0testplt[ipart,1], 0.0]
    
    
    for kph in arange(nph*nturntest):
        temp[kph+1, :] = temp[kph, :]
        fieldlines.timestep(temp[kph+1, :])
    yinttest[:,:,ipart] = temp
    # plt.plot( mod(temp[::nph,1], 2*pi), temp[::nph,0]*1e2, ',')

#plt.show(block = True)