from numpy import zeros, array, arange, append, hstack, pi, cos, sin, mod, linspace, concatenate,vstack

import numpy as np
from common import f, qe, c
import common
from fieldlines import fieldlines
import ghalton
import random


N = 70 # training data N = 70 for eps = 0.001
nphmap = 4 # number of splits
nm = 1000*nphmap
nturn = 2 # Number of full turns
nph = 100     # Number of steps per turn
mod_m = -3
mod_n = 2
eps =  0.001 
seq = ghalton.Halton(3)
X0 = seq.get(N)*array([0.38, 2*pi, 0])+array([0.1, 0, 0])#
yint = zeros([nph*nturn + 1, 3, N])

for ipart in range(0, N):
    
    r0 = X0[ipart, 0]
    th0 = X0[ipart, 1]
    ph0 = X0[ipart, 2]
    fieldlines.init(nph=nph, am=mod_m, an=mod_n, aeps=eps, aphase=0.0, arlast=r0)
    pth0 = qe/c*fieldlines.ath(r0, th0, ph0)

    z = zeros([nph*nturn + 1, 3])
    z[0,:] = [pth0, th0, 0.0]
    
    from time import time
    tic = time()
    
    for kph in arange(nph*nturn):
        z[kph+1, :] = z[kph, :]
        fieldlines.timestep(z[kph+1, :])
    yint[:,:,ipart] = z
    
#%%

print('Integration of training data done')
ind = int(nph/nphmap)

# organize training input data q = (N, nphmap), Q = (N, nphmap)
q = np.zeros([N, nphmap])
Q = np.zeros([N, nphmap])
p = np.zeros([N, nphmap])
P = np.zeros([N, nphmap])
for i in range(0, nphmap):
    q[:, i] = yint[int(i*ind), 1]
    p[:, i] = yint[int(i*ind), 0]*1e2
    Q[:, i] = yint[int((i+1)*ind), 1]
    P[:, i] = yint[int((i+1)*ind), 0]*1e2
    
# organize training output data zqtrain = (N, nphmap), zptrain = (N, nphmap)
zqtrain = Q - q
zptrain = p - P
ztrain = np.vstack((zptrain, zqtrain))
xtrain = np.vstack((q, P))
#%%

Ntest = 30
nturntest = int(nm/nphmap)
qminmap = 0.16
qmaxmap = 0.31 
pminmap = 0
pmaxmap = 2*np.pi

random.seed(1)
q0map = linspace(qminmap,qmaxmap,Ntest)
p0map = linspace(pminmap,pmaxmap,Ntest)
q0map = random.sample(list(q0map), Ntest)
p0map = random.sample(list(p0map), Ntest)
Q0map = np.array(q0map)
P0map = np.array(p0map)
X0test = np.stack((Q0map, P0map, np.zeros(Ntest))).T

start = time()
yinttest = zeros([nph*nturntest + 1, 3, Ntest])
print('Integration of reference data ...')
for ipart in range(0, Ntest):
    fieldlines.init(nph=nph, am=mod_m, an=mod_n, aeps=eps, aphase=0.0, arlast=X0test[ipart,0])
    X0test[ipart, 0] = qe/c*fieldlines.ath(X0test[ipart,0], X0test[ipart,1], X0test[ipart,2])
    
    temp = zeros([nph*nturntest + 1, 3])
    temp[0,:] = [X0test[ipart,0], X0test[ipart,1], 0.0]
    
    
    for kph in arange(nph*nturntest):
        temp[kph+1, :] = temp[kph, :]
        fieldlines.timestep(temp[kph+1, :])
    yinttest[:,:,ipart] = temp
yinttest[:,0] = yinttest[:, 0]*1e2
yinttest[:, 1] = yinttest[:, 1]
end = time()
print('Time reference data(high accuracy sympl. Euler): ', end-start)