import numpy as np
from pynfft.nfft import NFFT
from pynfft import Solver

# Number of uniform grid points
N =1000

# Number of non uniform "nodes"
M = 700

# The wavenumber of the eigenfunction we consider
k = -4

def make_eigenfunction( points, k ):
    '''
    returns exp( -2 PI i kx )
    '''
    return np.cos( 2j * np.pi * k * points )
    
    
def fourier_multiplier(f, eigs, plan, infft, eps = 1E-7 ):
    '''
    use an operator that is a Fourier multiplier
    '''
    # find Fourier coefficients. 
    infft.y = f
    infft.before_loop()
    
    while True:
        infft.loop_one_step()
        if(np.all(infft.r_iter < eps)):
            break    
            
    # use as a Fourier Multiplier
    f_hat = infft.f_hat_iter * eigs
        
    # transform back
    plan.f_hat = f_hat
    return plan.trafo()

def laplacian_eigenvalues( plan ):
    '''
    These are the eigenvalues of the negative laplacian
    they are stored as a grid correspoinding to the 
    wavenumber
    '''
    N = plan.N[0]
        
    # Seems like I need to divide by N. I have no idea why, though.
    k = np.arange( -N / 2, N / 2 ) / N 
    eigs =-(2*np.pi*k)**2
    return eigs

# initializations...
points = np.random.uniform( -0.5, 0.5, M) # randome points in the desired range 
plan  = NFFT( N, M ) # I have a PLAN
plan.x = points # set the points in the plan
plan.precompute() # we need to do this for some reason
infft = Solver( plan ) # solver finds fourier coefficients

# eigenvalues of laplacian with fourier basis
eigs = laplacian_eigenvalues( plan ) # eigenvalues of my 
f_eig = make_eigenfunction( points, k ) # an exponential is an egeinfunction of laplacian     

# Check that we can reconstruct the function
f_new = fourier_multiplier( f_eig, eigs**0, plan, infft, eps = 1E-16 )   
assert( np.allclose( f_eig, f_new ) )

# check that laplacian^2 = composition of laplacians
f1 = fourier_multiplier( f_eig, eigs**0, plan, infft, eps = 1E-9 )
f11= fourier_multiplier( f1   , eigs   , plan, infft, eps = 1E-9 )
f2 = fourier_multiplier( f_eig, eigs**2, plan, infft, eps = 1E-9 ) 
f1 = fourier_multiplier( f2   , eigs   , plan, infft, eps = 1E-9 )   
f3 = fourier_multiplier( f_eig, eigs**3, plan, infft, eps = 1E-9 )
assert( np.allclose( f11, f2  ) )  
assert( np.allclose( f3 , f21 ) )

# This is the *LAPLACIAN* eigenvalue associated with the eigenfunction
eigenvalue = -( 2*np.pi*k ) ** 2

# apply the laplacian to the eigenfunction
f_new = fourier_multiplier( f_eig, eigs1, plan, infft, eps = 1E-6 )
    
# this should have eigenvalue
ratios = np.divide( f_new , f_eig )
    
print( "Should be: " + str( eigenvalue ) )
print( "Is:\n"       + str( ratios     ) )
assert( np.allclose( f_new , f_eig ) )
