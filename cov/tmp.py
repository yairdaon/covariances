import numpy as np
from pynfft.nfft import NFFT
from pynfft import Solver
import matplotlib.pyplot as plt
import math
import os
os.system( "rm -rvf *.png")
# Number of uniform grid points
N = 10000

# Number of non uniform "nodes"
M = 500

# The wavenumber of the eigenfunction we consider
k = 3

def norm( v, pts ):
    assert( len( v ) == len( pts ) )
    n = len( v )
    
    h = pts[1:n-1] - pts[0:n-2]
    u = ( v[1:n-1] + v[0:n-2] ) / 2
    I = np.dot( h, u*np.conj(u) )
    return math.sqrt( I )

def normal( v, pts ):
    return v / norm( v, pts )

def make_rand_data( points ):
    '''
    iid N(0,1) data
    '''
    return np.random.normal( loc= 0.0, scale = 1, size = points.shape[0] )

def make_eigenfunction( points, k ):
    '''
    returns exp( -2 PI i kx )
    '''
    return np.cos(2 * np.pi * k * points )
    
       
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

def laplacian_eigenvalues( N ):
    '''
    These are the eigenvalues of the negative laplacian
    they are stored as a grid correspoinding to the 
    wavenumber
    '''        
    # Seems like I need to divide by N. I have no idea why, though.
    k = np.arange( -N / 2, N / 2 ) 
    eigs = 4 * np.pi * np.pi * k*k + 0.5
    return eigs

# initializations...
points = np.random.uniform( -0.5, 0.5, M-2) # randome points in the desired range 
points = np.concatenate( (points, np.array([-0.5,0.4999])) ) 
points = np.sort( points )
#points = np.arange( 0, 1+1/(M-1), 1/(M-1) )

plan = NFFT( N, M ) # I have a PLAN
plan.x = points # set the points in the plan
plan.precompute() # we need to do this for some reason
infft = Solver( plan ) # solver finds fourier coefficients


# eigenvalues of laplacian with fourier basis
eigs = laplacian_eigenvalues( plan.N[0] ) # eigenvalues of my 
f_eig = make_eigenfunction( points, k ) # an exponential is an egeinfunction of laplacian     
f_rand = make_rand_data( points )

inv_res = 1
i = 0


eigenvector = normal( f_eig, points ) 
lim = np.max( np.abs( eigenvector ) )
eigenvalue  = np.dot( fourier_multiplier( eigenvector, eigs, plan, infft, eps = 1E-9), eigenvector )
while True:
    
    
   
    tmp_eigs = np.power( eigenvalue - eigs, -1 )
    tmp = fourier_multiplier( eigenvector, tmp_eigs, plan, infft, eps = 1E-9)
    tmp = normal( tmp, points ) 
    inv_res = norm( tmp - eigenvector, points )  
    eigenvector = tmp
    eigenvalue  = np.dot( fourier_multiplier( eigenvector, eigs, plan, infft, eps = 1E-9), eigenvector ) / norm( eigenvector, points )**2   

    if i % 25 == 0:
        print("\n\n")
        print( i )
        print( inv_res )
        print( eigenvalue )
        plt.plot( points, eigenvector )
        axes = plt.gca()
        axes.set_ylim( [-lim, lim ] )
        plt.savefig( "inverse_tmp" + str( int( i / 25 ) ) + ".png")
        plt.close()  
        if inv_res < 1E-12:
            break

 
    i = i+1
    


    
