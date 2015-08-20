#from scipy.spatial import Delaunay
import numpy as np
from pynfft.nfft import NFFT
from pynfft import Solver


def make_exponential( points, k ):
    factor = -2j * np.pi
    prod = factor * np.einsum("ij , j -> i", points, k)
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
    f_hat = infft.f_hat_iter * eigs**power
    
    # transform back
    plan.f_hat = f_hat
    return plan.trafo()


def laplacian_eigenvalues( plan, eps = 0.5 ):
    dims = plan.N
    k_x = np.arange( -dims[1] / 2, dims[1] / 2 )
    k_y = np.arange( -dims[0] / 2, dims[0] / 2 )
    k_x, k_y = np.meshgrid( k_x, k_y )
    eigs = 4 * np.pi * np.pi * ( k_x * k_x + k_y * k_y ) + eps
    return eigs








# Points 
n_pts = 10
n_dims = 2
points = np.random.uniform( -0.5, 0.5, n_pts * n_dims )
points = points.reshape( n_pts , n_dims )


# create solver etc
n_grid = 500
plan  = NFFT( [ n_grid, n_grid ], points.shape[0] )
plan.x = points
plan.precompute()
infft = Solver( plan )
eigs = laplacian_eigenvalues( plan, eps = 1.0 )
f_rand = make_rand_data( points )
k = np.array([ -1,1 ])
f_exp = make_exponential( points, k )     

test = True
if not test:
 
    # Check that we can reconstruct the function
    f_new = fourier_multiplier( f_rand, eigs, 0, plan, infft, eps = 1E-9 )   
    assert( np.allclose( f_rand, f_new ) )

    f1 = fourier_multiplier( f_rand, eigs, 1, plan, infft, eps = 1E-9 )
    f11= fourier_multiplier( f1    , eigs, 1, plan, infft, eps = 1E-9 )
    f2 = fourier_multiplier( f_rand, eigs, 2, plan, infft, eps = 1E-9 )    
    assert( np.allclose( f11, f2 ) )  

if test:    
    
    eigenvalue = np.sum( k * k )
    f_new = fourier_multiplier( f_exp, eigs, 1, plan, infft, eps = 1E-14 )
    ratios = f_new / f_exp

    print( eigenvalue )
    print( np.mean( ratios ) / n_grid / n_grid )
    
    #p = 0.005
    #f_lap_inv = fourier_multiplier( rand_f, eigs, -p, plan, infft, eps = 1E-9 )
    #f_new = fourier_multiplier( f_lap_inv, eigs,  p, plan, infft, eps = 1E-9 )   
    #print( f_new - exp_f )
    #assert( np.allclose( f_new , rand_f ) )
 

if False:
    import matplotlib.pyplot as plt
    plt.plot(interior[:, 0], interior[ :,1], 'd' , color = 'g' )
    plt.plot(exterior[:, 0], exterior[ :,1], 'd' , color = 'b' )
    plt.plot(perimeter[:,0], perimeter[:,1], 'o' , color = 'r' )
    plt.title( "red = bdry, green = interior, blue = extrerior" )
    plt.show()
