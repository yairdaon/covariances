#!/usr/local/bin/python2.7
import numpy as np
import math
import scipy.sparse as sprs 
import scipy.sparse.linalg as la
import scipy.interpolate as polat
import matplotlib.pyplot as plt

import helper
import laplacian2d as lap
import container as cot

def multigrid_preconditioner( f, params ):
    
    assert f.shape == ( len( params.x_grid ) , len( params.y_grid ) ) 
    
    tmp    = cot.container( int( params.M / 2 ), #M
                            int( params.N / 2 ), #N
                            1E-6, #eps
                            helper.apply_inv_schur_comp,
                            params.alpha  , # alpha
                            params.sigma  , # sigma 
                            params.power  , # power
                            "Coarse" #grid_type
                            )
    

    # fine grid points
    X_f, Y_f = np.meshgrid( params.x_grid,  params.y_grid  )
    pts_f = np.vstack(  (np.ravel(X_f),np.ravel(Y_f))  )
    pts_f = pts_f.T
    assert pts_f.shape == (params.m * params.n, 2), str(pts_f.shape) +"\n != \n" + str( (params.m * params.n, 2) )
    
    # Coarse grid points
    X_c, Y_c = np.meshgrid( tmp.x_grid,  tmp.y_grid  )
    pts_c= np.vstack(  (np.ravel(X_c),np.ravel(Y_c))  )
    pts_c = pts_c.T
    assert pts_c.shape == (tmp.m * tmp.n, 2), str(pts_c.shape) +"\n != \n" + str( (tmp.m * tmp.n, 2) )

    # interpolant on FINE grid
    interpolant_f = polat.LinearNDInterpolator( pts_f , np.ravel(f), fill_value = 0.0 )
   
    # Interpolate to coarse grid
    f_c = interpolant_f( np.ravel(X_c), np.ravel(Y_c) )
    
    # Solve on COARSE grid
    coarse_solution, _ = la.gmres( tmp.cov, f_c, tol = tmp.eps, M = tmp.prec )
        
    # Interpolant to go back to fine grid
    interpolant_c = polat.LinearNDInterpolator( pts_c , np.ravel(coarse_solution), fill_value = 0.0 )
   
    fine_solution = interpolant_c( np.ravel(X_f), np.ravel(Y_f) )
    fine_solution = fine_solution.reshape( (params.n,params.m) )

    assert f.shape == fine_solution.shape
    print( fine_solution )
    
    return fine_solution
    
def apply_precision( f, params ):
    """
    We' like to apply an inverse covariace to f. 
    However, it is not that simple - we don't have
    direct access to the inverse covariacne using
    pads and projections. We can apply covariacne
    and apply the inverse of its schur complement.
    
    The former is the target of the conjugate 
    gradients method. The latter is used as a
    preconditioner.
    """
    # A chunk of code that defines the callback function
    ####################################################
    def gen(): # generator that counts:
        cur = 0
        while True:
            yield cur
            cur = cur + 1    
    def callback( x, A, gen, grid_type ): # do these instructions every iteration:
        counter = gen.next()
        if counter % 25 == 0:
            residual = np.linalg.norm( A(np.ravel(x)) - np.ravel(f) )
            print( grid_type + " grid. CG cycles = " + str( counter ) + ", r = " + str( residual ) + "." )
   
    generator = gen()
    wrapper = lambda x: callback( x, params.cov, generator, params.grid_type   ) # just a wrapper
    ####################################################

    solution, _ = la.cg( params.cov, np.ravel(f), tol = params.eps, callback = wrapper , M = params.prec )
    return solution.reshape( (params.n,params.m) )
                           
if __name__ == "__main__":
    
    params = cot.container( 16, #M
                            16, #N
                            1E-8 , #eps
                            multigrid_preconditioner,
                            #helper.apply_inv_schur_comp, 
                            0.2, # alpha
                            500, #sigma 
                            -1.1, #power
                            "Fine" #grid_type
                            )

    # Apply covariance and precision #########
    f = lap.make_f( params.m  , params.n   ) * params.sigma  
    initial_residual = np.linalg.norm( np.ravel(f) )
    params.eps = params.eps * initial_residual
    cov_f = helper.apply_covariance( f, params )
    reconstruct_f = apply_precision( cov_f, params ) 
    err = helper.norm( f - reconstruct_f )
    print( "Reconstruction Error = " + str( err ) )

    # Plot eigenvalues
    # if False:
    #     lin_op = la.LinearOperator(
    #         ( params.in_size   , params.in_size   ) , 
    #         lambda v: np.ravel( apply_covariance( v.reshape( params.in_dims   ) , params ) )
    #         )
    #     eigs = la.eigs( lin_op, k = 1000, return_eigenvectors = False ) 
    #     eigs = np.sort( eigs )
    #     eigs = eigs[0:-1 
    #     eigs = eigs[::-1 
    #     k = np.arange( 1, len( eigs ) + 1) 
    #     powa = 1.04
    #     k = k**powa
    #     eigs = eigs * k
        
    #     plt.plot( eigs, 'ro' )    
    #     plt.xlabel('$k$')
    #     plt.ylabel('$\lambda_k \cdot k^{' + str(powa) + '}$')
    #     plt.title( 'Eigenvalues of 2D covariance after cancelling their decay' )
    #     plt.subplots_adjust(left=0.15)
    #     plt.savefig( "eigenvalues.png" )
    #     plt.close()
       
    # # Sample from the gaussian ############## 
    for i in range( 10 ):
        smp = helper.sample( params )
        
        # plot - boring...
        plt.imshow( smp )#,vmin = -1, vmax = 1)
        plt.colorbar()
        plt.title( params.desc + "\n" )
        plt.savefig( "Sample " + str(i) + ".png" )
        plt.close()
    # Unfortunately, this takes waaaay too much time
    # Plot empirical covariance matrix ####### 
    # cov_matrix = np.zeros( (m*n, m*n) )
    # num_samples = 10
    # for i in range( num_samples ):
    #     if i % 1 == 0:
    #         print( str(i) +"th sample" )
    #         plt.imshow( cov_matrix / num_samples )
    #         print("imshow")
    #         plt.colorbar()
    #         print("colorbar")
    #         plt.title( "Covariance matrix of " + cov_str +"\nusing " + str(i) + " samples." )
    #         print("title")
    #         plt.savefig( "Temporary Covariance.png" )
    #         print("savefig")
    #         plt.close()
    #         print( "close" )
            
                        
    #     smp = sample( params )
    #     smp = np.ravel( smp )
    #     cov_matrix = cov_matrix + np.outer( smp, smp )

    # # Plot. BS
    # plt.imshow( cov_matrix / num_samples )
    # plt.colorbar()
    # plt.title( "Covariance matrix of " + cov_str +"\nusing " + str(i) + " samples." )
    # plt.savefig( "Covariance Matrix.png" )
    # plt.close()
    

    # The neumann!!!
    # X = np.linspace( 0, 1, num = params.m' , endpoint = True )
    # Y = np.linspace( 0, 1, num = params.n' , endpoint = True )
    # X, Y = np.meshgrid( X, Y )
    
    # f = -4 * np.pi * np.pi * np.cos( 2 * np.pi * X )
    # u = apply_with_neumann( f, params )
    # u = u - np.mean( u )
    # plt.imshow( u )
    # plt.colorbar()
    # plt.savefig( "solved u.png" )
    # plt.close()
    # v = np.cos( 2 * np.pi * X ) 
    # v = v - np.mean( v ) 
    # plt.imshow( v )
    # plt.colorbar()
    # plt.savefig( "true u.png" )
    # plt.close()
    # #print( v /  u )
    # #assert False

# def apply_with_neumann( u, params ):
#     '''
#     use neumann bdry conditions. make sure the condition
#     \int \int_{\Omega} f = 
#     \ind_{\partial \Omega} \frac{\partial u}{\partial n } 
#     holds
#     '''    
#     h = params.hx' 
#     m = params.m' 
#     n = params.n' 
    
#     # Here we determine a Neumann BC that is consistent
#     # with u, as stated in the docstring
#     int_u = np.sum( u ) * h * h
#     bdry_size = ( 2*m + 2*n - 2 ) * h
#     c = int_u / bdry_size
#     correction = 2.0 * h * c
     
#     # The basic averaging scheme of the laplacian. 
#     # every point is -4 * u_point plus:
#     # sum over neighbours of u_neighbour
#     just_sum = -4*u
#     just_sum[0:-1, :  ] = just_sum[0:-1, :    + u[1:   , :  ]
#     just_sum[1:  , :  ] = just_sum[1:  , :    + u[0:-1 , :  ]
#     just_sum[ :  ,0:-1] = just_sum[ :  ,0:-1] + u[ :   ,1:  ]
#     just_sum[ :  ,1:  ] = just_sum[ :  ,1:  ] + u[ :   ,0:-1]

#     # edges...
#     just_sum[   0 , 1:-1 ] = just_sum[   0 , 1:-1 ] + u[ 1    , 1:-1 ] + correction
#     just_sum[  -1 , 1:-1 ] = just_sum[  -1 , 1:-1 ] + u[-2    , 1:-1 ] + correction
#     just_sum[1:-1 ,    0 ] = just_sum[1:-1 ,    0 ] + u[ 1:-1 ,    1 ] + correction
#     just_sum[1:-1 ,   -1 ] = just_sum[1:-1 ,   -1 ] + u[ 1:-1 ,   -2 ] + correction
    
#     # corners
#     just_sum[ 0 , 0 ] = just_sum[ 0 , 0 ] + u[ 1 , 0 ] + u[ 1 , 0 ] + math.sqrt(2) * correction
#     just_sum[ -1, 0 ] = just_sum[-1 , 0 ] + u[-2 , 0 ] + u[ 1 , 1 ] + math.sqrt(2) * correction
#     just_sum[ 0 ,-1 ] = just_sum[ 0 , -1] + u[ 1 , -1] + u[ 0 , -2] + math.sqrt(2) * correction
#     just_sum[-1 ,-1 ] = just_sum[-1 , -1] + u[-2 , -1] + u[-1 , -2] + math.sqrt(2) * correction
 
#     # make sure we use POSITIVE DEFINITE LAPLACIAN
#     just_sum = -just_sum / (h*h)

#     # multiply by beta
#     just_sum = params.beta  * just_sum

#     # add alpha times identity...
#     just_sum = just_sum + params.alpha * f

#     #c'est tout
#     return just_sum
