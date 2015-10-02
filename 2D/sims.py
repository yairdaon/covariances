#!/usr/local/bin/python3
import numpy as np
import matplotlib.pyplot as plt

import cg
import laplacian2d as lap
from covar2d import *
 
grids = [ 40, 100 , 200, 400, 800, 1600 ]
fmt = [ "%4.1f" ,    "%3.2f",   "%1.7f",    "%3.2f" ,   "%3.2f" ,   "%3.2f" ]
print( str( fmt ) )
header = "grid, CG cycles,     error,       eps,      power,     alpha"
data = np.empty( 6 )
data[0] = 2
data[1] = 0.3
data[2] = 0.3
data[3] = 0.3
data[4] = 0.3
data[5] = 0.3

# for i in range( len(grids) ):

#     M = grids[i]
#     N = grids[i]

#     # Preparations
#     m = int(M / 4) # small domain grid pts  in x direction
#     n = int(N / 4) # small domain grid pts  in y direction

#     # Use this to pad with zeros. See routine "pad".
#     big_zeros = np.zeros( ( N, M ) )
               
#     # Define laplacian-like operator as -(1-alpha)*Laplacian + alpha * Id
#     alpha = 0.025 

#     # Negative fractional power used to define the **COVARIANCE**
#     power = -1.13

#     # Scalar, used to scale
#     sigma = 1

#     # Generate eigenvalues laplacian like operator
#     eigs = lap.laplacian_eigenvalues( M, N, alpha )

#     # Eigenvalues of the covariance over the entire domain
#     cov_eigs      = sigma**2      * np.power( eigs, power     )

#     # Eigenvalues of cov to half power. Used for sampling a Gaussian
#     cov_half_eigs = sigma         * np.power( eigs, power / 2 )

#     # Eigenvalues of the precision operator over entire domain
#     prec_eigs     = sigma**(-2)   * np.power( eigs, -power    )  
    
#     # threshold used in conjugate gradients method
#     eps = 1E-7
 
#     # Parameters that we'll carry around
#     params = {}
#     params['cov_eigs'        ] = cov_eigs
#     params['cov_half_eigs'   ] = cov_half_eigs 
#     params['prec_eigs'       ] = prec_eigs
#     params['big_zeros'       ] = big_zeros   
#     params['M'               ] = M
#     params['N'               ] = N    
#     params['m'               ] = m
#     params['n'               ] = n
#     params['eps'             ] = eps

#     count = 0
#     error = 0.0
#     n_it = 15
#     for j in range( n_it ):

#         f = lap.make_f( m, n ) * sigma
#         cov_f = apply_covariance( f, params )
#         reconstruct_f, new_count = apply_precision( cov_f, params, get_count = True )
#         count = count + new_count
#         error = error + lap.norm( f - reconstruct_f )    
        
#     count = count / n_it
#     error = error / n_it
#     data[0] = M
#     data[1] = count
#     data[2] = error
#     data[3] = eps
#     data[4] = power
#     data[5] = alpha
print( "baam" )
with open("data.txt",'a') as f_handle:
    np.savetxt( f_handle, data, fmt = fmt, delimiter = "      " , comments = "", header = header)
