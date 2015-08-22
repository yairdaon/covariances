#from scipy.spatial import Delaunay
import numpy as np
from pynfft.nfft import NFFT
from pynfft import Solver

# choose which "test" to run
test = True

# Points 
n_dims = 1
n_grid =900
n_pts = 50
N = np.ones( n_dims ) * n_grid
alpha = 1.0
k = np.array([ 1 ])

def make_exponential( points, k ):
    
    factor = -2j * np.pi
    #prod = factor * np.einsum("ij , j -> i", points, k)
    prod = factor * points * k 
    ret = np.exp( prod )
    return ret 

def make_rand_data( points ):
    '''
    iid N(0,1) data
    '''
    return np.random.normal( loc= 0.0, scale = 1, size = points.shape[0] )

def fourier_multiplier(f, eigs, power, plan, infft, eps = 1E-7 ):
    '''
    use an operator that is a Fourier multiplier
    '''
    # find Fourier coefficients. Called "solve" in this context
    infft.y = f
    infft.before_loop()
    
    while True:
        infft.loop_one_step()
        if(np.all(infft.r_iter < eps)):
            break    

    #assert( eigs.shape == infft.f_hat_iter.shape )

    # use as a Fourier Multiplier
    f_hat = infft.f_hat_iter * ( eigs**power )
    
    # transform back
    plan.f_hat = f_hat
    return plan.trafo()

def laplacian_eigenvalues( plan, alpha = 0.5 ):
    '''
    These are the eigenvalues of the laplacian
    they are stored as a grid correspoinding to the 
    frequencies
    '''
    dims = plan.N

    if hasattr( dims, "size" ) and dims.size == 2:
        k_x = np.arange( -dims[1] / 2, dims[1] / 2 ) / plan.N[1]
        k_y = np.arange( -dims[0] / 2, dims[0] / 2 ) / plan.N[0]
        k_x, k_y = np.meshgrid( k_x, k_y )
        eigs = 4 * np.pi * np.pi * ( k_x**2 + k_y**2 ) + alpha
    else:
        k = np.arange( -dims[0] / 2, dims[0] / 2 ) / plan.N[0]
        eigs = k**2 + alpha
    return eigs

# initializations...
points = np.random.uniform( -0.5, 0.5, n_pts * n_dims ) # randome numbers 
points = points.reshape( n_pts , n_dims ) # reshape to get rand points
plan  = NFFT( N, points.shape[0] ) # I have a PLAN
plan.x = points # set the points in the plan
plan.precompute() # we need to do this for some reason
infft = Solver( plan ) # solver finds fourier coefficients
eigs = laplacian_eigenvalues( plan, alpha = alpha ) # eigenvalues of my operator
f_rand = make_rand_data( points ) # random data
f_exp = make_exponential( points, k ) # an exponential is an egeinfunction of laplacian     

if test:
 
    # Check that we can reconstruct the function
    f_new = fourier_multiplier( f_rand, eigs, 0, plan, infft, eps = 1E-16 )   
    assert( np.allclose( f_rand, f_new ) )

    # check that laplacian^2 = composition of laplacians
    f1 = fourier_multiplier( f_rand, eigs, 1, plan, infft, eps = 1E-9 )
    f11= fourier_multiplier( f1    , eigs, 1, plan, infft, eps = 1E-9 )
    f2 = fourier_multiplier( f_rand, eigs, 2, plan, infft, eps = 1E-9 )    
    assert( np.allclose( f11, f2 ) )  


    
    power = 1 

    # This is the *LAPLACIAN* eigenvalue associated with the exponential
    eigenvalue = np.sum( k * k )

    
    # apply the laplacian to the eigenfunction
    f_new = fourier_multiplier( f_exp, eigs, power, plan, infft, eps = 1E-34 )

    # this should have eigenvalue + alpha
    ratios = f_new / f_exp

    print( "Should be: " + str(         ( eigenvalue + alpha )**power ) )
    print( "Is:        " + str(  np.mean( ratios             )        ) )
    
    #p = 0.005
    #f_lap_inv = fourier_multiplier( rand_f, eigs, -p, plan, infft, eps = 1E-9 )
    #f_new = fourier_multiplier( f_lap_inv, eigs,  p, plan, infft, eps = 1E-9 )   
    #print( f_new - exp_f )
    #assert( np.allclose( f_new , rand_f ) )
 

if False:
    import matplotlib.pyplot as plt
    plt.plot(points[:,0], points[:,1], 'o' , color = 'r' )
    plt.show()
