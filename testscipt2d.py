import numpy as np
from matplotlib import pyplot as plt
from seminar_03 import guided_modes_2D
import time

plt.close('all')

grid_size     = [100,100]     # please choose appropriate value
number_points = 100*100     # please choose appropriate value
h             = 0.005   # please choose appropriate value
lam           = 0.78
k0            = 2*np.pi/lam
e_substrate   = 1.5**2
delta_e       = 1.5e-2
w             = 15.0
numb          = 10

time_start = time.time()
prm2 = epsilon_gauss(grid_size, h , e_substrate, delta_e, w) #calculate permittivity profile
eigval2, eigvec2=guided_modes_2D(prm2, k0, h,numb) #calculate eigenmodes

time_end = time.time() 
time_c= time_end - time_start #Time for calculation

#calculate the coordinates
y = h*(np.linspace(1,grid_size[0],grid_size[0])-round(grid_size[0]/2))
x = h*(np.linspace(1,grid_size[1],grid_size[1])-round(grid_size[1]/2))
X, Y = np.meshgrid(x,y)

#plot 3D image
#from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(figsize=[20,20])
# ax = Axes3D(fig)
# ax.plot_surface(X, Y, abs(eigvec2[0,:,:])/np.max(abs(eigvec2[0,:,:])), rstride=1, cstride=1, cmap='rainbow')
# plt.show()

#plot and save 2D images
for i in range(0,numb):
    fig2=plt.figure()
    plt.imshow(abs(eigvec2[i,:,:]),extent=[x[-1], x[0], y[-1], y[0]])  # if we want to normalize the fields: /np.max(abs(eigvec2[i,:,:]))
    cbar=plt.colorbar()
    plt.xlabel('x/$\mu m$')
    plt.ylabel('y/$\mu m$')
    plt.title('eigenvalue = {0}'.format(eigval2[i]))
    plt.show()
    fig2.savefig('100_0.05_order{0}.png'.format(i),dpi=720) #save the image

