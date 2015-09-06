import numpy as np
from numpy.fft import rfft as rfft
from numpy.fft import irfft as irfft
import matplotlib.pyplot as plt
import math

# Number of times we call Fourier Multiplier
count = 0

def dot( u, v ):
    '''
    We do a dot product that approximates
    the L2 (funciton space) dot product.
    We ignore imaginary parts since all we 
    do we do in real domains
    '''
    n = len( u ) 
    zeros = np.zeros( n )
    assert np.allclose( u.imag , zeros )
    assert np.allclose( v.imag , zeros )
    return np.dot( u.real, v.real ) / n

def norm( u ):
    '''
    Approximate the L2 norm in function space!

    '''
    return math.sqrt( dot( u, u ) ) 

def fourier_multiplier( f, eigs ):
    '''
    use an operator that is a Fourier multiplier
    '''
    # How many times we use this Fourier multiplier?
    global count
    count = count + 1


    assert np.allclose( f.imag , np.zeros ( len( f ) ) )
    f_hat = rfft( f )
    
    assert len( f_hat ) == len( eigs ), str( len( f_hat ) ) + "\n" + str( len ( eigs )  )
    f_hat = f_hat * eigs
        
    
    return irfft( f_hat )    

def laplacian_eigenvalues( N, alpha ):
    '''
    These are the eigenvalues of the laplacian-like
    operator. 

    We take a convex combination of the identity operator
    and the laplacian-like operator. The coefficient of
    the identity is alpha and the coefficient of the
    laplacian-like operator is 1-alpha = beta.
    '''
    # Evenly numbered domains are your friend.
    assert( N % 2 == 0 )

    # Alpha is the relative weight we give the 
    # identity operator. 
    assert( alpha > 0 and alpha < 1 )

    # This is the weight we give the laplacian-like
    # operator.
    beta = 1 - alpha

    # What we get when twice differnetiating complex exponentials
    factor = 4 * np.pi * np.pi 

    eigs = np.arange( 0, N/2 + 1 ) 
    assert len( eigs ) == N/2 + 1 , str( len(eigs) ) + "\n" + str( N/2+1 )
    
    return alpha + factor * eigs * eigs * beta 


def make_f( N ):
    '''
    iid N(0,1) data
    '''
    factors = np.arange( 0, N/2 + 1 ) + 1
    factors[0:15] = 1
    f_hat_real = np.random.normal( loc = 0.0, scale = 1, size = N/2+1 ) / factors**3.2 
    f_hat_imag = np.random.normal( loc = 0.0, scale = 1, size = N/2+1 ) / factors**3.2
    f = irfft( f_hat_real + 1j * f_hat_imag, N )
    assert len( f ) == N
    return f #/ norm( f )

if __name__ == "__main__":

    # Erase previous run
    import os
    os.system( "rm -rvf *.png")

    # Number of uniform grid points
    N = 500
    points = np.linspace( -0.5,  0.5, N ,endpoint = False )
    
    # eigenvalues of laplacian with fourier basis
    alpha = 0.25
    beta = 1 - alpha
    eigs = laplacian_eigenvalues( N, alpha ) 
    f = make_f( N )
    eigenvector = f / norm( f )  
    eigenvalue = dot( fourier_multiplier( eigenvector, eigs ), eigenvector )
    wavenumber = math.sqrt(  (eigenvalue-alpha) / beta  ) / np.pi / 2 
    approx = np.min( np.abs( eigs - eigenvalue ) )
    i = 0
    while approx > 1E-10:  
        
        # Plot previous
        plt.plot( points, eigenvector )
        axes = plt.gca()
        axes.set_ylim( [-2, 2 ] )
        axes.set_xlim( [ -0.5,0.5 ] )
        title = "Estimates: $k = %.2f$, Error $= %.4f$" % ( wavenumber, approx )
        plt.title( title )
        plt.savefig( "Rayleigh Quotient Iteration " + str( i ) + ".png")
        plt.close()  
    
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

        eigenvalue = dot( fourier_multiplier( eigenvector, eigs ), eigenvector )  
        wavenumber = math.sqrt( (eigenvalue - alpha)/beta ) / np.pi / 2  

        approx = np.min( np.abs( eigs - eigenvalue ) )
        i = i + 1
    
    # Plot Last
    plt.plot( points, eigenvector )
    axes = plt.gca()
    axes.set_ylim( [-2, 2 ] )
    axes.set_xlim( [ -0.5,0.5 ] )
    title = "Estimates: $k = %.2f$, Error $= %.4f$" % ( wavenumber, approx )
    plt.title( title )
    plt.savefig( "Rayleigh Quotient Iteration " + str( i ) + ".png")
    plt.close()   
    
    
    
    

 
    


    

