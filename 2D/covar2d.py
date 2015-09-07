#!/usr/local/bin/python3
import numpy as np
import matplotlib.pyplot as plt

import cg
import laplacian2D as lap

def project( v, params ):
    '''
    Take a vector v of lenght N and return its projection
    to the first n coordinates.
    
    In Linear Algebraic terms, project can be thought
    of as a n x N matrix A s.t. A_{ij} = 1 iff i = j and 
    A_{ij} = 0 otherwise. It is the transpose of the 
    pad matrix.
    '''
    pass

def pad( v , params ):
    '''
    take a vector v of length n and pad it with zeros
    so that the resulting vector has length N.

    In Linear Algebraic terms, pad can be thought of as
    a N x n matrix A s.t. A_{ij} = 1 iff i = j and 
    A_{ij} = 0 otherwise. It is the transpose of the 
    projection matrix.
    '''    
    pass

def sample( params ):
    '''
    Sample a gaussian with mean zero and covariance 
    operator which is defined using an eigenvalue
    decomposition in Fourier domain.
    '''    
    # Generate iid gaussians on a mesh corresponding to
    # the ENTIRE domain
    Z = np.random.normal( size =  params['big_domain'] )  

    # Apply covariance to power half. Makes this a sample from
    # our covaraince function.
    sample = lap.fourier_multiplier( Z, params['cov_half_eigs'] )
    
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
    g = pad( f, params )

    cov_g = lap.fourier_multiplier( g, params['cov_eigs'] )        
    
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
    g = pad( f, params )

    cov_g = lap.fourier_multiplier( g, params['prec_eigs'] )
      
    return project( cov_g, params )
  
def apply_precision( f, params ):
    ''' 
    We'd like to apply an inverse covariace to f. 
    However, it is not that simple - we don't have
    direct access to the inverse covariacne using
    pads and projections. We can apply covariacne
    and apply the inverse of its schur complement.
    
    The former is the target of the conjugate 
    gradients method. The latter is used as a
    preconditioner. We access both using the same
    routien, only with a different flag - namely 
    the inv_schur flag.
    '''      
    
    x_0 = np.zeros( params['small_domain'] )
    return cg.pcg( f,
                   x_0,
                   apply_covariance,
                   params,
                   apply_inv_schur_comp,
                   params,
                   params['eps'] )

    
# Set the points
big_domain = 6000  # Total domain
small_domain = big_domain / 2 # only the inside
leftover = big_domain - small_domain

# Eigenvalues of the laplacian-like operator, no power involved yet.
alpha = 0.25 
power = -1.13
sigma = 1

eigs = lap.laplacian_eigenvalues( big_domain, alpha )
cov_eigs      = sigma**2      * np.power( eigs, power     )
cov_half_eigs = sigma         * np.power( eigs, power / 2 )
prec_eigs     = sigma**(-2)   * np.power( eigs, -power    )  


# threshold for conjugate gradients
eps = 1E-5
 
# Parameters that we'll carry around
params = {}
params['cov_eigs'        ] = cov_eigs
params['cov_half_eigs'   ] = cov_half_eigs 
params['prec_eigs'       ] = prec_eigs

params['cov_power'       ] = power
params['big_domain'      ] = big_domain
params['small_domain'    ] = small_domain
params['eps'             ] = eps


if __name__ == "__main__":
    
    # Description of covariance operator
    cov_str = "$(  %.2f \Delta +  %.2fI)^{ %.2f}$" % (alpha-1, alpha, power)
    reconstruct_str = "$(  %.2f \Delta +  %.2fI)^{ %.2f} \circ $ " % (alpha-1, alpha, -power)
    reconstruct_str = reconstruct_str + cov_str

    # Sample from the gaussian ##############
    bound = 0
    for i in range( 20 ):
        smp = sample( params )
        bound = max( bound, np.max( np.abs( smp ) ) ) 
        
        # plot - boring...
        plt.plot( inside, smp )
    axes = plt.gca()
    axes.set_ylim( [-1.2 * bound, 1.2 * bound ] )
    axes.set_xlim( [ 0.25,0.75 ] )
    plt.title( "Samples from Gaussian with zero mean and covaraince " + cov_str +
               ".\nBig domain is $[0,1]$ interval. Subdomain is $[0.25,0.75]$." )
    plt.savefig( "Samples.png" )
    plt.close()

    # Apply the covariance and its inverse ##########
    lap.count = 0
    print( lap.count )
    f = lap.make_f( len(inside) ) * sigma
    cov_f = apply_covariance( f, params )
    reconstruct_f = apply_precision( cov_f, params )
    print( lap.count )
    
    # Plot shit 
    plt.plot( inside, reconstruct_f, color = "b" , label = "reconstructed u" )
    plt.plot( inside, cov_f        , color = "r" , label = cov_str + "u"     )
    plt.plot( inside, f            , color = "g" , label = "u"               )
    axes = plt.gca()
    axes.set_xlim( [ 0.25,0.75 ] )
    plt.title( "Apply " + cov_str + "\nand its inverse to reconstruct to a function." )
    plt.legend( loc=2, prop={'size':6} )
    plt.savefig( "Apply Covariance and precision.png" )
    plt.close()

    # Plot empirical covaraince matrix #############
    num_samples = 50000
    cov_matrix = 0
    for i in range( num_samples ):
        smp = sample( params )
        smp = project( smp, params ) 
        cov_matrix = cov_matrix + np.outer( smp, smp )

    cov_matrix = cov_matrix / num_samples
    plt.imshow( cov_matrix )
    plt.colorbar()
    plt.title( "Covariance matrix arising by truncating " + cov_str +" using "
               +str(num_samples) + " samples." )
    plt.savefig( "Covariance Matrix.png" )
    plt.close()
