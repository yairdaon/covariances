#!/usr/local/bin/python3
import numpy as np
import laplacian2d as lap

def dot( A, B ):
    return np.dot( np.ravel(A),np.ravel(B) )

def pcg( b, x, apply_A, params_A, solve_M, params_M, eps, get_count = False, print_count = False ):
    '''
    Preconditioned Conjugate Gradients method 
    from Wikipedia

    v_c is current v, v_pr is previous v etc. 
    '''
    count = 0
    
    # Here k == 0
    x_c = x # Initial guess
    r_c = b - apply_A( x_c , params_A ) # Intial residual
    z_c = solve_M( r_c, params_M ) # Initial z
    p_c = z_c # Initial other thing
    
    while True:
        count = count + 1
        
        
        # Save some computation cuz we use this value twice
        Ap_c = apply_A( p_c, params_A )

        alpha = dot( r_c, z_c ) / dot( p_c, Ap_c )

        # Update x and discard previous value
        x_c = x_c + alpha * p_c
        
        # Update r and keep previous value
        r_pr = r_c
        if count % 20 == 0:
            r_c = b - apply_A( x_c, params_A )
        else:
            r_c = r_pr - alpha * Ap_c  
        
        res = lap.norm( r_c )  
        if print_count and count % 50 == 0:
            print( "CG, " + str(count) +"th iteration. Residual == " + str( res ) )  
          
        if res < eps:
            if get_count:
                return ( x_c, count )
            else:
                return x_c

        # Keep previous value and then update
        z_pr = z_c
        z_c = solve_M( r_c, params_M )

        beta = dot( r_c, z_c ) / dot( r_pr, z_pr )
        
        # Overwrite, won't need previous valuex
        p_c = z_c + beta * p_c

    
if __name__ == "__main__":

    # n, as in R^n
    n = 500
 
    # Symmetric positive definite A
    A = np.random.normal( size = n*n ).reshape( (n,n) )
    A = np.dot( A.T, A ) + 0.5
    
    # Diagonal preconditioner
    D = np.diagonal( A )

    # Define these using functions cuz that's what our routine takes
    apply_A = lambda x, params_A: np.dot( A, x )
    solve_M = lambda x, params_M: x / D
    
    # Target vector
    b = np.random.normal( size = n )
    
    # Initial guess
    x = np.random.normal( scale = 50, size = n ) 
    
    # Parameters
    eps = 1e-10 # threshold for stopping condition
    params_A = { "param" : 4 }  # BS arguments for A
    params_M = { "parameter" : 3 }  # BS arguments for M
    
    
    # Solve!
    x = pcg( b, x, apply_A, params_A, solve_M, params_M, eps )
    
    # Calcualte residual r = Ax - b
    r = np.linalg.norm( apply_A( x, params_A ) - b )
    print( "r_k = " + str( r ) )

