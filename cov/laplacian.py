import numpy as np
from numpy.fft import rfft as rfft
from numpy.fft import irfft as irfft
import matplotlib.pyplot as plt
import math
import os
os.system( "rm -rvf *.png")


def dot( u, v ):
    n = len( u ) 
    zeros = np.zeros( n )
    assert np.allclose( u.imag , zeros )
    assert np.allclose( v.imag , zeros )
    return np.dot( u.real, v.real ) / n

def norm( u ):
    return math.sqrt( dot( u, u ) ) 

def fourier_multiplier( f, eigs ):
    '''
    use an operator that is a Fourier multiplier
    '''
    assert np.allclose( f.imag , np.zeros ( len( f ) ) )
    f_hat = rfft( f )
    
    assert len( f_hat ) == len( eigs ), str( len( f_hat ) ) + "\n" + str( len ( eigs )  )
    f_hat = f_hat * eigs
        
    return irfft( f_hat )    

def laplacian_eigenvalues( N, alpha ):
    '''
    These are the eigenvalues of the negative laplacian
    they are stored as a grid correspoinding to the 
    wavenumber
    '''
    assert( N % 2 == 0 )
    factor = 4 * np.pi * np.pi 

    eigs = np.arange( 0, N/2 + 1 ) 
    assert len( eigs ) == N/2 + 1 , str(len(eigs) ) + "\n" + str(N/2+1)
    
    return alpha + eigs * eigs * factor


def make_f( N ):
    '''
    iid N(0,1) data
    '''
    factors = np.arange( 0, N/2 + 1 ) + 1
    f_hat_real = np.random.normal( loc = 0.0, scale = 1, size = N/2+1 ) /factors
    f_hat_imag = np.random.normal( loc = 0.0, scale = 1, size = N/2+1 ) /factors
    f = irfft( f_hat_real + 1j * f_hat_imag, N )
    assert len( f ) == N
    return f

if __name__ == "__main__":
    # Number of uniform grid points
    N = 500
    points = np.linspace( -0.5,  0.5, N ,endpoint = False )
    
    # eigenvalues of laplacian with fourier basis
    alpha = 0.1
    eigs = laplacian_eigenvalues( N, alpha ) 
    f = make_f( N )
    eigenvector = f / norm( f )  
    eigenvalue = dot( fourier_multiplier( eigenvector, eigs ), eigenvector )
    approx = 1
    i = 0
    while approx > 1E-10:  
        print()
        print( i )
        plt.plot( points, eigenvector )
        axes = plt.gca()
        axes.set_ylim( [-2, 2 ] )
        axes.set_xlim( [ -0.5,0.5 ] )
        plt.savefig( "inverse_basix" + str( i ) + ".png")
        plt.close()  
    
        inverse_eigenvalues = np.power( eigs - eigenvalue, -1 )
        new = fourier_multiplier( eigenvector, inverse_eigenvalues )
        new = new / norm( new )  
        res1 = norm( new - eigenvector ) 
        res2 = norm( new + eigenvector ) 
        if res1 > res2:
            eigenvector = -new 
            res = res2
        else:
            eigenvector = new
            res = res1

        eigenvalue = dot( fourier_multiplier( eigenvector, eigs ), eigenvector )  
        wavenumber = math.sqrt( eigenvalue - alpha ) / np.pi / 2  
        approx = np.min( np.abs( eigs - eigenvalue ) )
        print( "wavenumber    = " + str( wavenumber ) )
        print( "approximation = " + str( approx ) ) 
        i = i + 1
    
    
    
   
    
    
    
    

 
    


    

