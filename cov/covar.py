import laplacian as lap
import numpy as np
import matplotlib.pyplot as plt
def project( v, intervals ):
    '''
    Take a vector v of lenght N and return its projection
    to the first n coordinates.
    
    In Linear Algebraic terms, project can be thought
    of as a n x N matrix A s.t. A_{ij} = 1 iff i = j and 
    A_{ij} = 0 otherwise. It is the transpose of the 
    pad matrix.
    '''
    start = len( intervals[0] )
    stop  = len( intervals[1] ) + start
    return v[ start:stop ]

def pad( v , intervals ):
    '''
    take a vector v of length n and pad it with zeros
    so that the resulting vector has length N.

    In Linear Algebraic terms, pad can be thought of as
    a N x n matrix A s.t. A_{ij} = 1 iff i = j and 
    A_{ij} = 0 otherwise. It is the transpose of the 
    projection matrix.
    '''    
    left_zeros  = np.zeros( len( intervals[0] ) )
    right_zeros = np.zeros( len( intervals[2] ) ) 
    return np.concatenate( (left_zeros, v, right_zeros ) )

def sample( N, eigs, power ):
    '''
    Sample a gaussian with mean zero and covariance 
    operator which is
    
    Laplacian^{-power}

    The division of power by two is due to taking 
    square root of covaraince matrix
    '''
    Z = np.random.normal( size = N )
    covariance_half_eigs = np.power( eigs, power / 2 ) 
    return lap.fourier_multiplier( Z, covariance_half_eigs )
     
  
def apply_covariance( f, eigs, power, intervals ):
    ''' 
    f is thought of as a function in the SMALLER
    domain. Hence we need to pad it with zeros
    and project the result
    '''
    g = pad( f, intervals )
    eigs_to_power = np.power( eigs, power )
    cov_g = lap.fourier_multiplier( g, eigs_to_power )
    cov_f = project( cov_g, intervals )
    return cov_f
    
def apply_precision( f, eigs, power, intervals ):
    ''' 
    f is thought of as a function in the SMALLER
    domain. Hence we need to pad it with zeros
    and project the result
    '''
    g = pad( f, intervals )
    laplacian_g = lap.fourier_multiplier( g, eigs )
    laplacian_f = project( laplacian_g, intervals )
    return laplacian_f
    
# parameterssss
N = 6000  # Total domain
n = N / 2 # only the inside

alpha = 0.1
power = -.8

# Eigenvalues of the laplacian-like operator, no power involved yet.
eigs = lap.laplacian_eigenvalues( N, alpha )

# Set the points
inside , in_step = np.linspace( 0.25, 0.75, n        , endpoint = False , retstep = True ) 
left   , l_step  = np.linspace( 0   , 0.25, (N-n)/2  , endpoint = False , retstep = True )
right  , r_step  = np.linspace( 0.75, 1   , (N-n)/2  , endpoint = False , retstep = True )
assert in_step == l_step and l_step == r_step, str( in_step ) + "  " + str( l_step ) + "   " + str( r_step )

intervals = [ left, inside, right ]

pts = np.concatenate( ( inside, left, right ) )


# Sample from the gaussian
bound = 0
for i in range( 20 ):
    smp = sample( N, eigs, power )
    smp = project( smp, intervals  ) 
    bound = max( bound, np.max( np.abs( smp ) ) ) 
    plt.plot( inside, smp )
axes = plt.gca()
axes.set_ylim( [-1.2 * bound, 1.2 * bound ] )
axes.set_xlim( [ 0.25,0.75 ] )
cov_str = str("$\Delta^{" + str(power) + "}$")
plt.title( "Samples from Gaussian with zero mean and covaraince " + cov_str +".\nBig domain is $[0,1]$ interval. Subdomain is $[0.25,0.75]$." )
plt.savefig( "Samples.png" )
plt.close()

# Apply the covariance function
f = lap.make_f( len(inside) )
laplacian_f = apply_covariance( f, eigs, power, intervals )
plt.plot( inside, laplacian_f , color = "r" , label = cov_str + "u" )
plt.plot( inside, f           , color = "g" , label = "u"           )
axes = plt.gca()
axes.set_xlim( [ 0.25,0.75 ] )
plt.title( "Apply " + cov_str + " to a function." )
plt.legend()
plt.savefig( "Apply Covariance.png" )
plt.close()

# Plot the covaraince matrix
num_samples = 5000
cov_matrix = 0
for i in range( num_samples ):
    smp = sample( N, eigs, power )
    smp = project( smp, intervals  ) 
    cov_matrix = cov_matrix + np.outer( smp, smp )

cov_matrix = cov_matrix / num_samples
plt.imshow( cov_matrix )
plt.colorbar()
plt.title( "Covaraince matrix arising by truncating " + cov_str +" using " +str(num_samples) + " samples." )
plt.savefig( "Covariance Matrix.png" )
plt.close()
