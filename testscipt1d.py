import numpy as np
import time
from matplotlib import pyplot as plt
from seminar_03 import guided_modes_1DTE,epsilon_gauss


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

time_start = time.time() #Timer begins
eigval, eigvec=guided_modes_1DTE(prm, k0, h)
time_end = time.time()   #Timer ends
 
 
time_c= time_end - time_start

legend1=list()
for val in eigval:
    legend1.append("eigenvalue {:.5f}".format(val))
    


fig = plt.figure(figsize=[10,10]) 
ax=fig.add_subplot(111)
x=h*(np.linspace(1,grid_size,grid_size)-round(grid_size/2))
ax.plot(x,eigvec)
ax2 = ax.twinx()
ax2.plot(x,prm,'--')
ax.set_position([0.2, 0.1, 0.8, 0.8])
ax.legend(legend1,loc='best')
ax.set(xlabel = 'x/$\mu m$', ylabel = 'E/a.u.')
ax2.set(ylabel='permittivity')
plt.title('Eigenmodes')
plt.tight_layout()
plt.savefig('{0}_{1}_1D.png'.format(grid_size,h))
plt.show()
