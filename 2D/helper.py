import numpy as np
import scipy.sparse.linalg as la
import math

import laplacian2d as lap

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

def project( v, params ):
    '''
    Take a vector v and return its projection
    to the coordinaes we care for.
    ''' 
    return v[ 0:params.n, 0:params.m ] 

def pad( v , params ):
    '''
    Take a vector v of length n and pad it with zeros
    so that the resulting vector has length N.

    In Linear Algebraic terms, pad can be thought of as
    a N x n matrix A s.t. A_{ij} = 1 iff i = j and 
    A_{ij} = 0 otherwise. It is the transpose of the 
    projection matrix.
    '''    
    big_zeros = params.big_zeros
    big_zeros[ 0:params.n, 0:params.m ] = v
    return big_zeros

def sample( params ):
    '''
    Sample a gaussian with mean zero and covariance 
    operator which is defined using an eigenvalue
    decomposition in Fourier domain.
    '''    
    # Generate iid gaussians on a mesh corresponding to
    # the ENTIRE domain
    size = params.M * params.N
    Z = np.random.normal( size = size ) 
    Z = Z.reshape( ( params.M, params.N ) )

    # Apply covariance to power half. Makes this a sample from
    # our covaraince function.
    sample = lap.fourier_multiplier( Z, params.cov_half_eigs )
    
    # throw away unnecessary part of the sample
    return project( sample, params ) 

def apply_covariance( f, params ):
    ''' 
    Apply covariance in a subdomain.

                   [ C_inside    |  C_boundary  ] 
    Define C_all = [ ------------|------------- ],
                   [ C_boundary' |  C_outside   ]
    
    the covaraince matrix used on the entire domain.
    
    This routine returns  C_inside * f.
    '''
    # Pad with zeros
    g = pad( f, params )

    # Apply big Covariance
    cov_g = lap.fourier_multiplier( g, params.cov_eigs )        
    
    return project( cov_g, params )
 
def apply_inv_schur_comp( f, params ):
    ''' 
    Apply inverse of covariance's Schur complement.

                   [ C_inside    |  C_boundary  ] 
    Define C_all = [ ------------|------------- ],
                   [ C_boundary' |  C_outside   ]
    
    the covaraince matrix used on the entire domain.
    
    This routines returns  
    (C_in - C_bd * C_out^{-1} * C_bd' )^{-1} * f. 
    This is the inverse of the schur complement of
    C_in applied to f.
    '''
    assert f.shape == ( len( params.x_grid ) , len( params.y_grid ) )
    
    g = pad( f, params )

    cov_g = lap.fourier_multiplier( g, params.prec_eigs )
      
    return project( cov_g, params )
