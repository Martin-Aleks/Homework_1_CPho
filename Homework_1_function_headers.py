'''Homework 1, Computational Photonics, SS 2020:  FD mode solver.
'''
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigs


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
    
    M = np.zeros((300, 300))
    for i in range(M[0][:].size):
        if i == 0: # boundary
            M[i][i] = -2/(h**2)+k0**2*prm[i]
            M[i][i+1] = 1/(h**2)
        elif i>0 and i<299:
            M[i][i] = -2/(h**2)+k0**2*prm[i]   # central diagonal
            M[i][i+1] = 1/(h**2)
            M[i][i-1] = 1/(h**2)
        elif i == 299:
            M[i][i] = -2/(h**2)+k0**2*prm[i]
            M[i][i-1] = 1/(h**2)
    
    in_put = 1/(k0^2)*M # multiplication according to the formula
    eigval, eigvev = LA.eig(in_put) # now getting the eigenvalues (eff eps) and eigenvectors (fields)
    print(eigval)
    print(eigvec)
    # now need to select fields that satisfy the conditions 
    
    pass


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





