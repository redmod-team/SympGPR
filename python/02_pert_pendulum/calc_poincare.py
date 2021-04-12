import numpy as np
import ghalton
import time
nm = 100 #map applications
e = 0.5
om = 0.5
def zdot(t, z): #perturbed pendulum
    x = z[:,0:1]
    y = z[:,1:]
    xdot=e*(0.3*x*np.sin(2*t) + 0.7*x*np.sin(3*t))+y
    ydot=-e*(0.3*y*np.sin(2*t) + 0.7*y*np.sin(3*t)) - om**2*np.sin(x)
    return np.hstack([xdot,ydot])

#%% see Burby et al. (2020): Fast neural Poincaré maps for toroidal magnetic fields
# Los Alamos technical report LA-UR-20-24873
def gen_samples_circle(origin, radius, n_samples):
    s_radius = .5*radius**2
    seq = ghalton.Halton(2)
    samp = seq.get(n_samples)*np.array([s_radius, 2*np.pi])
    s = samp[:,0]
    theta = samp[:,1]
    x = origin[0]+np.sqrt(2*s)*np.cos(theta)
    y = origin[1]+np.sqrt(2*s)*np.sin(theta)
    return np.hstack([x.reshape(n_samples,1),y.reshape(n_samples,1)])

def gen_samples_pmap(origin,r1,nics,n_iterations,eps):
    rkstep=1500
    latent_samples = gen_samples_circle(origin, r1,nics)
    sample = latent_samples
    out = np.zeros([nics,2])
    for i in range(n_iterations):
        tempbin,sample = rk_pmap(sample,eps,rkstep)
        out[:,:] = sample[:,:]
    return [out,latent_samples]


#%%
def rk_pmap(z,eps,n_rk_steps = 100):
    dphi = 2*np.pi/n_rk_steps
    phi_current = 0
    z_current = 1.0*z
    out = np.zeros([N, 2, n_rk_steps])
    for i in range(n_rk_steps):
        k1 = zdot(phi_current, z_current)
        k2 = zdot(phi_current + .5*dphi,z_current + .5*dphi*k1)
        k3 = zdot(phi_current + .5*dphi,z_current + .5*dphi*k2)
        k4 = zdot(phi_current + dphi, z_current + dphi * k3)
        z_current = z_current + (1.0/6.0) * dphi * (k1 + 2*k2 + 2*k3 + k4)
        phi_current = phi_current + dphi
        out[:,:,i] = z_current
    return out, z_current

N = 55
radius=0.9
[labels_raw, data_raw] = gen_samples_pmap([0,0], radius, N, 1, e)
rr=(labels_raw[:,0])**2+labels_raw[:,1]**2
ind=np.argwhere(rr<=radius**2)
n_data=len(ind)
data=np.zeros([n_data,2])
labels=np.zeros([n_data,2])
data[:,0:1]=data_raw[ind,0]
data[:,1:2]=data_raw[ind,1]
labels[:,0:1]=labels_raw[ind,0]
labels[:,1:2]=labels_raw[ind,1]

q = data[:, 0] + np.pi
p = data[:, 1]
Q = labels[:,0] + np.pi
P = labels[:,1]
    
zqtrain = Q - q
zptrain = p - P

xtrain = q.flatten()
ytrain = P.flatten()
xtrain = np.hstack((q, P)).T
ztrain1 = zptrain.flatten()
ztrain2 = zqtrain.flatten()
ztrain = np.concatenate((ztrain1, ztrain2))

N = n_data
print('Training data: ', N)

#%%

## Test data for perturbed pendulum
nics = 20
Ntest = int(nics+nics//2)
xic = np.linspace(0.05,0.7,nics).reshape([nics,1])
yic = 0.0*np.ones([nics,1])
yic2 = np.linspace(0.3,0.6,nics//2).reshape([nics//2,1])
xic2 = 0.0*np.ones([nics//2,1])
zic = np.hstack([np.vstack([xic,xic2]),np.vstack([yic,yic2])])
qs = zic[:,0]
ps = zic[:,1]

#%%
start = time.time()
yinttest = np.zeros([2,Ntest,nm])
for i in range(Ntest):
    temp = np.array([zic[i]])
    yinttest[:,i,0] = temp
    for k in range(nm-1):
        tempbin, temp = rk_pmap(temp, e, 100)
        yinttest[:,i,k+1] = temp
end = time.time()
print('Time needed RK: ', end-start)

# transform by + np.pi for symplectic GP map
qs = zic[:,0]+np.pi
ps = zic[:,1]
yinttest[0] = yinttest[0]+np.pi
