from numpy import zeros, array, arange, append, hstack, pi, cos, sin, mod, linspace, concatenate,vstack
import numpy as np
from common import f, qe, c
import common
from fieldlines import fieldlines
import ghalton
import random
#
# Attention: Variables are [pth, th, ph], so momentum in first component and
#            "time" in third. Change to usual Z = z[:,1::-1] before using!
#
N = 70 # training data N = 70 for eps = 0.001
nm = 500
nturn = 2 # Number of full turns
nph = 32     # Number of steps per turn

nphmap = 4
eps =  0.001 
seq = ghalton.Halton(3)
X0 = seq.get(N)*array([0.38, 2*pi, 0])+array([0.1, 0, 0])#
yint = zeros([nph*nturn + 1, 3, N])

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

print('finish')
ind = int(nph/nphmap)

q1 = yint[0,1]
p1 = yint[0,0]*1e2
Q1 = yint[ind,1]
P1 = yint[ind,0]*1e2

q2 = Q1
p2 = P1
Q2 = yint[int(2*ind),1]
P2 = yint[int(2*ind),0]*1e2

q3 = Q2
p3 = P2
Q3 = yint[int(3*ind),1]
P3 = yint[int(3*ind),0]*1e2

q4 = Q3
p4 = P3
Q4 = yint[int(4*ind),1]
P4 = yint[int(4*ind),0]*1e2

#%%
zqtrain1 = Q1 - q1
zptrain1 = p1 - P1
xtrain1 = q1.flatten()
ytrain1 = P1.flatten()
xtrain1 = hstack((q1, P1)).T
ztrain1 = concatenate((zptrain1.flatten(), zqtrain1.flatten()))

zqtrain2 = Q2 - q2
zptrain2 = p2 - P2
xtrain2 = q2.flatten()
ytrain2 = P2.flatten()
xtrain2 = hstack((q2, P2)).T
ztrain2 = concatenate((zptrain2.flatten(), zqtrain2.flatten()))

zqtrain3 = Q3 - q3
zptrain3 = p3 - P3
xtrain3 = q3.flatten()
ytrain3 = P3.flatten()
xtrain3 = hstack((q3, P3)).T
ztrain3 = concatenate((zptrain3.flatten(), zqtrain3.flatten()))

zqtrain4 = Q4 - q4
zptrain4 = p4 - P4
xtrain4 = q4.flatten()
ytrain4 = P4.flatten()
xtrain4 = hstack((q4, P4)).T
ztrain4 = concatenate((zptrain4.flatten(), zqtrain4.flatten()))

Ntest = 30
nturntest = int(nm/4)
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

yinttest = zeros([nph*nturntest + 1, 3, Ntest])
for ipart in range(0, Ntest):
    fieldlines.init(nph=nph, am=-3, an=2, aeps=eps, aphase=0.0, arlast=X0test[ipart,0])
    X0test[ipart, 0] = qe/c*fieldlines.ath(X0test[ipart,0], X0test[ipart,1], X0test[ipart,2])
    
    temp = zeros([nph*nturntest + 1, 3])
    temp[0,:] = [X0test[ipart,0], X0test[ipart,1], 0.0]
    
    
    for kph in arange(nph*nturntest):
        temp[kph+1, :] = temp[kph, :]
        fieldlines.timestep(temp[kph+1, :])
    yinttest[:,:,ipart] = temp
yinttest[:,0] = yinttest[:, 0]*1e2
yinttest[:, 1] = yinttest[:, 1]