import numpy as np
from matplotlib import pyplot as plt
from seminar_03 import guided_modes_1DTE


plt.close('all')

grid_size     = 300 # please choose appropriate value
#number_points = 300       # please choose appropriate value
h             = 0.05      # please choose appropriate value
lam           = 0.78
k0            = 2*np.pi/lam
e_substrate   = 1.5**2
delta_e       = 1.5e-2
w             = 15.0

prm = epsilon_gauss(grid_size, h , e_substrate, delta_e, w)
eigval, eigvec=guided_modes_1DTE(prm, k0, h)
