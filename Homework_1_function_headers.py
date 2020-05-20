'''Homework 1, Computational Photonics, SS 2020:  FD mode solver.
'''
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sps
from scipy.sparse.linalg import eigs


def epsilon_gauss(eps_size, h , eps_sub, Delta_eps, W):
    """Computes the permittivity profile ε(x,ω)=ε_sub+Δε*exp[-(x/W)^2]
       All dimensions are in µm.
    
    Parameters
    ----------
    eps_size : 1d-array with 1 or 2 integers 
    
        Number of sampling points along x (or x and y) direction 
        
    h : float
    
        Spatial discretization
    
    eps_sub : float
        
        permittivity of the substrate
        
    Delta_eps : float
        
        a parameter of the Gaussian waveguide profile
        
    W : float
        
        a parameter of the Gaussian waveguide profile
    
    
    Returns
    -------
    D : defined by the argument size
    
        permittivity profile
    
        
        
    """
    D = np.zeros(eps_size)
    
    
    if np.size(eps_size) == 1  :
        x = h*(np.linspace(1,eps_size,eps_size)-round(eps_size/2))
        D = eps_sub+Delta_eps*np.exp(-(x/W)**2)
        # computes only for x direction
    
    elif np.size(eps_size) == 2:
        x = h*(np.linspace(1,eps_size[0],eps_size[0])-round(eps_size[0]/2))
        y = h*(np.linspace(1,eps_size[1],eps_size[1])-round(eps_size[1]/2))
        X, Y = np.meshgrid(x,y)
        R2 = X**2+Y**2
        D = eps_sub+Delta_eps*np.exp(-R2/W**2)
        #computes for x-y plane
    else:
        raise ValueError('Invalid input: '
                         'the size of the profile is not a 1d or 2d array')
    
    return D


def guided_modes_1DTE(prm, k0, h):
    """Computes the effective permittivity of a TE polarized guided eigenmode.
    All dimensions are in µm.
    Note that modes are filtered to match the requirement that
    their effective permittivity is larger than the substrate (cladding).
    
    Parameters
    ----------
    prm : 1d-array
        Dielectric permittivity in the x-direction
    k0 : float
        Free space wavenumber
    h : float
        Spatial discretization
    
    Returns
    -------
    eff_eps : 1d-array
        Effective permittivity vector of calculated modes
    guided : 2d-array
        Field distributions of the guided eigenmodes
    """
    #####################################################
    # Trying to build the Matrix in slide 5
    # Tested with simple numbers and works fine. Can provide test code.
    
    N=np.size(prm)
    M = np.zeros([N,N],prm.dtype)

    for i in range(N):
        if i == 0: 
            # boundary
            M[i][i] = -2/(h**2)+k0**2*prm[i]
            M[i][i+1] = 1/(h**2)
        elif i>0 and i< (N-1):
            M[i][i] = -2/(h**2)+k0**2*prm[i]  
            # central diagonal
            M[i][i+1] = 1/(h**2)
            M[i][i-1] = 1/(h**2)
        elif i == N-1:
            M[i][i] = -2/(h**2)+k0**2*prm[i]
            M[i][i-1] = 1/(h**2)
            
    in_put = 1/(k0**2)*M # multiplication according to the formula
    eigval, eigvev = LA.eig(in_put) # now getting the eigenvalues (eff eps) and eigenvectors (fields)
    #print(eigval)
    #print(eigvec)
    # now need to select fields that satisfy the conditions 
    eigval_eff=eigval[(min(prm)<eigval)&(eigval<max(prm))]
    eigvec_eff=eigvec[:][(min(prm)<eigval)&(eigval<max(prm))]                              
    return eigval_eff, eigvec_eff
    


def guided_modes_2D(prm, k0, h, numb):
    """Computes the effective permittivity of a quasi-TE polarized guided 
    eigenmode. All dimensions are in µm.
    
    Parameters
    ----------
    prm  : 2d-array
        Dielectric permittivity in the xy-plane
    k0 : float
        Free space wavenumber
    h : float
        Spatial discretization
    numb : int
        Number of eigenmodes to be calculated
    
    Returns
    -------
    eff_eps : 1d-array
        Effective permittivity vector of calculated eigenmodes
    guided : 3d-array
        Field distributions of the guided eigenmodes
    """
    pass





