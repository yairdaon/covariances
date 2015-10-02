import numpy as np
import scipy.sparse.linalg as la

import helper
import laplacian2d as lap

class container():
    
    def __init__( self,
                  M,
                  N,
                  eps,
                  preconditioner,
                  alpha,
                  sigma,
                  power,
                  grid_type ):
   
    
        assert alpha >= 0 and alpha < 1 
        assert power < 0
        
        
        # Stuff we fix
        self.alpha          = alpha # Define laplacian-like operator as -(1-alpha)*Laplacian + alpha * Id
        self.beta           = 1 - alpha
        self.sigma          = sigma # Scalar, used to scale
        self.power          = power # Negative fractional power used to define the **COVARIANCE**
        self.M              = M # big domain grid points in x direction
        self.N              = N # big domain grid points in y direction    
        self.eps            = eps # threshold used in conjugate gradients method    
        self.preconditioner = preconditioner
        self.grid_type      = grid_type
        
        # Stuff we derive
        self.m              = m = int( M / 2 ) # small domain grid pts  in x direction
        self.n              = n = int( N / 2 ) # small domain grid pts  in y direction
        self.hx             = hx = 1.0 / M 
        self.hy             = hy = 1.0 / N
        self.x_grid         = np.arange( 0, hx * m, hx )
        self.y_grid         = np.arange( 0, hy * n, hy )
        self.big_zeros      = np.zeros((N,M)) # Use to pad with zeros. See routine "pad".
        if alpha > 0:
            self.desc           = "$\mathcal{C} = %4d^{2} \cdot  %.2f(-\Delta )^{%.2f} +  %.2f$ (Domain: $\int_{[0,1]^2} u = 0$)." % (sigma, 1-alpha, power, alpha)    
        else:
            self.desc           = "$\mathcal{C} = %4d^{2} (-\Delta )^{%.2f}$ (Domain: $\int_{[0,1]^2} u = 0$)." % (sigma, power)    
        # Generate eigenvalues laplacian like operator
        eigs = lap.laplacian_eigenvalues( M, N ) 
        eigs[0] = 1 # just so we can invert, eventually we set it to zero
        
        # Eigenvavalues of covariance over entire domain
        self.cov_eigs =  alpha + (1-alpha) * sigma**2 * np.power(eigs,power ) 
        
        # Eigenvalues of square root of covariance. used for sampling a Gaussian
        self.cov_half_eigs  = np.power( self.cov_eigs, 1 / 2.0 ) 

        # Eigenvalues of precision over entire domain
        self.prec_eigs      = 1 / self.cov_eigs 
        
        # Enforce zero mean:
        self.cov_eigs[0] = 0
        self.cov_half_eigs[0] = 0
        self.prec_eigs[0] = 0

        # LinearOperator types for scipy.sparse.linalg.cg 
        self.cov  = la.LinearOperator( (m*n,m*n), lambda x: helper.apply_covariance( x.reshape((n,m)), self ) )
        self.prec = la.LinearOperator( (m*n,m*n), lambda x: preconditioner         ( x.reshape((n,m)), self ) )

        
