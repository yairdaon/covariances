#!/usr/local/bin/python3

import numpy as np
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from numpy.fft import rfft2 as rfft2
from numpy.fft import irfft2 as irfft2

import matplotlib.pyplot as plt
import math

def dot( u, v ):
    '''
    We do a dot product that approximates
    the L2 (funciton space) dot product.
    We ignore imaginary parts since all we 
    do we do in real domains
    '''
    u = np.ravel( u )
    v = np.ravel( v )
    assert np.all( u.imag == 0 )
    assert np.all( v.imag == 0 )
    return np.dot( u, v ) / len( u ) 

def norm( u ):
    '''
    Approximate the L2 norm in function space!
    '''
    return math.sqrt( dot( u, u ) ) 

def fourier_multiplier( f, eigs ):
    '''
    use an operator that is a Fourier multiplier
    '''
    # Let's not deal with complex numbers just now....
    assert np.all( f.imag == 0 )
    
    # Do FFT, duuhh!!!
    f_hat = rfft2( f )
    
    assert f_hat.shape == eigs.shape, "\n" + str( f_hat.shape ) + "\n" + str( eigs.shape )
    
    # A fourier multiplier does this:
    f_hat = f_hat * eigs
        
    # Leave frequecncy domain...
    f_new = irfft2( f_hat )
    
    return f_new.real    

def laplacian_eigenvalues( M, N, alpha ):
    '''
    These are the eigenvalues of the laplacian-like
    operator. 

    We take a convex combination of the identity operator
    and the laplacian-like operator. The coefficient of
    the identity is alpha and the coefficient of the
    laplacian-like operator is 1-alpha = beta.
    '''
    # Evenly numbered domains are your friend.
    assert( M % 2 == 0 )
    assert( N % 2 == 0 )
    
    # Alpha is the relative weight we give the 
    # identity operator. 
    assert( alpha > 0 and alpha < 1 )

    # This is the weight we give the laplacian-like
    # operator.
    beta = 1 - alpha

    # What we get when twice differnetiating complex exponentials
    factor = 4 * np.pi * np.pi 

    eigs_x = np.arange( 0, M/2 + 1 ) 
    eigs_y = np.arange( 0, N ) 
    eigs_x, eigs_y = np.meshgrid( eigs_x, eigs_y )

    eigs = eigs_x * eigs_x + eigs_y * eigs_y
    #assert eigs.shape == ( M/2 + 1, N ) , "\n" + str( eigs.shape ) + "\nis not\n" + str( ( M/2 + 1, N ) )

    return alpha + beta * factor * eigs 

def make_f( M, N ):
    '''
    iid N(0,1) data
    '''
    k = np.arange( 0, N ) + 1
    l = np.arange( 0, M ) + 1
    
    k = 1. / k / k / k
    l = 1. / l / l / l
    k , l = np.meshgrid( k, l ) 

    k_real = np.reshape( np.random.normal( size = k.size ), ( M, N ) ) * k
    k_imag = np.reshape( np.random.normal( size = k.size ), ( M, N ) ) * k
    l_real = np.reshape( np.random.normal( size = l.size ), ( M, N ) ) * l 
    l_imag = np.reshape( np.random.normal( size = l.size ), ( M, N ) ) * l 
    
    f = ifft2( k_real + 1j * k_imag + l_real + 1j * l_imag, ( M, N ) )
    
    f = f.real
    return f / norm( f )

def find_wavenumbers( f ):
    M, N = f.shape
    f_hat = rfft2( f )
    return np.unravel_index( f_hat.argmax(), f_hat.shape )
    
    
if __name__ == "__main__":

    # Erase previous run
    import os
    os.system( "rm -rvf Rayleigh*.png")

    # Number of uniform grid points
    N = 256
    M = 256
    
    # eigenvalues of laplacian with fourier basis
    alpha = 0.25
    beta = 1 - alpha
    eigs = laplacian_eigenvalues( M, N, alpha ) 

    # Intial function
    eigenvector = make_f( M, N ) 
    i = 0
    while True:

        eigenvalue = dot( fourier_multiplier( eigenvector, eigs ), eigenvector )
        k2l2 = ( eigenvalue-alpha ) / beta / np.pi / np.pi / 4
        approx = np.min( np.abs( eigs - eigenvalue ) )
        k, l = find_wavenumbers( eigenvector )
        
        # Plot previous
        plt.imshow( eigenvector )
        plt.colorbar()
        title = "Wavenumbers: $k^2 + l^2 = %.1f, k = %.2f, l = %.2f$.\n Eigenvalue Error $= %.3f$" % ( k2l2, k, l, approx )
        plt.title( title )
        plt.savefig( "Rayleigh Quotient Iteration " + str( i ) + ".png")
        plt.close()  

        if approx < 1e-5:
            break

        # Update
        inverse_eigenvalues = np.power( eigs - eigenvalue, -1 )
        new = fourier_multiplier( eigenvector, inverse_eigenvalues )
        new = new / norm( new )  
        
        # Take care of everything else
        res1 = norm( new - eigenvector ) 
        res2 = norm( new + eigenvector ) 
        if res1 > res2:
            eigenvector = -new 
            res = res2
        else:
            eigenvector = new
            res = res1

        i = i + 1

 
    


    

