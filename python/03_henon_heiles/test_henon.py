import numpy as np
import matplotlib.pyplot as plt
import henon

N = 10

# Setting module options
henon.henon.lam = 0.5
henon.henon.tmax = 1000.0

out = []

# Tracing orbits
for ipart in range(N):
  z0 = np.random.rand(4) - 0.5
  z0[0] = 0.0
  tcut, zcut, icut = henon.integrate(z0)
  out.append([tcut[:icut], zcut[:,:icut]])
  plt.plot(out[-1][1][1,:], out[-1][1][3,:], '.')
