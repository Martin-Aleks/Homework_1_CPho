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
        y = h*(np.linspace(1,eps_size[0],eps_size[0])-round(eps_size[0]/2))
        x = h*(np.linspace(1,eps_size[1],eps_size[1])-round(eps_size[1]/2))
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
    eigval, eigvec = LA.eig(in_put) # now getting the eigenvalues (eff eps) and eigenvectors (fields)
    #print(eigval)
    #print(eigvec)
    # now need to select fields that satisfy the conditions 
    eigval_eff=eigval[(min(prm)<eigval)&(eigval<max(prm))]
    eigvec_eff=eigvec[:,(min(prm)<eigval)&(eigval<max(prm))]                              
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
    
    # Building the big matrix for N^2 by N^2 for an N by N grid of points
    H,W = prm.shape
    prm = prm.flatten()
    # Building the big matrix for h*w by h*w for an h by w grid of points
    #Calculate the elements along 4 diagnal lines
    arr0 = np.ones(H*W)*(-4) #

    arr1 = np.ones((H*W)-1)

    arr2 = np.ones(W*(H-1))

    for i in range(0, (H-1)): 

        arr1[(W-1)+(i*W)]=0

    arr0 = arr0/(k0**2)/(h**2) + prm

    arr1 = arr1/(k0**2)/(h**2)

    arr2 = arr2/(k0**2)/(h**2)
    #Construct the maxtrix M
    data = (arr2, arr1, arr0, arr1, arr2)

    M = sps.diags(data, [-W, -1, 0, 1, W])

    #select the proper number of eigenvalues
    N_cal = 0 #Number of eigenmodes that have been calculated
    e_min = min(prm) #limitation of effective permittivity
    e_max = max(prm) #limitation of effective permittivity
    val_eff1 = np.zeros(numb,complex) #store the eigenvalues
    vec_eff1 = np.zeros([H*W,numb],complex) #store the eigenvectors
    
    while N_cal<numb: 
        #when we still need to calculate more eigenvaluess that satisfy the limitation
        #notice that lower order modes have larger effective permittivity
        vals,vecs=eigs(M,k=2*numb,sigma=e_max) #calculate eigenvalues around e_max (the eigenvalues are arranged according to the value of abs(vals-sigma) from small to large )
        eigval1=vals[(e_min<vals)&(vals<e_max)]  #then choose the modes that satisfy the limitation
        eigvec1=vecs[:,(e_min<vals)&(vals<e_max)] 
        if (eigval1.size!=0):
            #when there are some modes that satisfy the limitation
            if (numb-N_cal<=eigval1.size):
                #if the calculation result provides enough number of modes we want
                #choose the number of modes we need
                val_eff1[N_cal:numb]=eigval1[0:(numb-N_cal)]
                vec_eff1[:,N_cal:numb]=eigvec1[:,0:(numb-N_cal)]
                N_cal=numb #update N_cal
            elif (numb-N_cal>eigval1.size):
                #if we still need to do more calculation
                #first store the results we have got
                val_eff1[N_cal:(N_cal+eigval1.size)]=eigval1[:]
                vec_eff1[:,N_cal:(N_cal+eigval1.size)]=eigvec1[:,:]
                N_cal=N_cal+eigval1.size #update N_cal to start next iteration
                e_max=min(val_eff1) #update the e_max to make sure that we don't select the modes that have been calculated(larger eigenvalues have been calculated)
        else:
            #when all the eigenvalues that satisfy the limitation have be calculated
            #but we didn't get enough number of modes we want
            str='There are not enough allowed eigenmodes in the structure.'\
            'The total number of modes that can exist is {0}'.format(N_cal)
            warnings.warn(str)
            break
        
    if N_cal==0:
        #if there is no mode that satisfies the limitation, raise an error
        raise ValueError('There exits no eigenmodes.')
    else:
        #reshape the size of eigenvectors
        vec_eff1=vec_eff1[:,0:N_cal]
        vec_eff1=vec_eff1.T
        vec_eff1=vec_eff1.reshape(N_cal,H,W)

    return val_eff1[0:N_cal],vec_eff1





