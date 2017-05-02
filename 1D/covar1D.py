#!/usr/local/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la
import pdb

import scipy.sparse.linalg as la
from scipy.sparse.linalg import LinearOperator as Linear
import laplacian1D as lap


class Parameters(object):
    
    def __init__( self, size=6000, alpha=0.25, power=-0.78, sigma=1 ):

        self.alpha = alpha
        self.power = power
        self.sigma = sigma
        
        # Set the points
        self.big_domain = size  # Total domain
        self.small_domain = self.big_domain // 2 # only the inside
        leftover = self.big_domain - self.small_domain

        self.inside , in_step = np.linspace( 
            0.25, 0.75, self.small_domain , endpoint = False , retstep = True ) 
        
        self.left   , l_step  = np.linspace(
            0   , 0.25, leftover / 2 , endpoint = False , retstep = True )
        
        self.right  , r_step  = np.linspace(
            0.75, 1   , leftover / 2 , endpoint = False , retstep = True )
        
        assert in_step == l_step 
        assert l_step == r_step
        #, str( in_step ) + "  " + str( l_step ) + "   " + str( r_step )
        
        self.pts = np.concatenate( ( self.inside, self.left, self.right ) )

        self.eigs = lap.laplacian_eigenvalues( self.big_domain, alpha )
        self.cov_eigs      = sigma**2      * np.power( self.eigs, self.power     )
        self.cov_half_eigs = sigma         * np.power( self.eigs, self.power / 2 )
        self.prec_eigs     = sigma**(-2)   * np.power( self.eigs, -self.power    )  
      
        
        # threshold for conjugate gradients
        self.eps = 1E-5

    def project( self, v ):
        '''
        Take a vector v of lenght N and return its projection
        to the first n coordinates.
        
        In Linear Algebraic terms, project can be thought
        of as a n x N matrix A s.t. A_{ij} = 1 iff i = j and 
        A_{ij} = 0 otherwise. It is the transpose of the 
        pad matrix.
        '''
                
        start = len( self.left )
        stop  = len( self.inside ) + start
        
        if len(v.shape) == 1:
            return v[ start:stop ]
        else:
            return v[ start:stop,: ]


    def pad( self, v ):
        '''
        take a vector v of length n and pad it with zeros
        so that the resulting vector has length N.
        
        In Linear Algebraic terms, pad can be thought of as
        a N x n matrix A s.t. A_{ij} = 1 iff i = j and 
        A_{ij} = 0 otherwise. It is the transpose of the 
        projection matrix.
        '''    
       
        if len(v.shape) == 1:
            v = v.reshape(len(v),1)
        n_samples = v.shape[1]
        
        left_zeros  = np.zeros( (len(self.left) , n_samples) )
        right_zeros = np.zeros( (len(self.right), n_samples) ) 

        return np.concatenate( (left_zeros, v, right_zeros ), axis = 0 )

    def sample( self, n_samples ):
        '''
        Sample a gaussian with mean zero and covariance 
        operator which is
        
        Laplacian^{-power}
    
        The division of power by two is due to taking 
        square root of covaraince matrix
        
        m is the number of samples to generate
        '''
        
        # Generate iid gaussians on a mesh corresponding to
        # the ENTIRE domain
        Z = np.random.randn( self.big_domain,n_samples )  
        
        # Apply covariance to power half. Makes this a sample from
        # our covaraince function.
        sample = lap.fourier_multiplier( Z, self.cov_half_eigs )
        
        return sample 

    def apply_covariance( self, f ):
        ''' 
        Apply covariance in a subdomain.
        
        [ C_inside    |  C_boundary  ] 
        Define C_all = [ ------------|------------- ],
        [ C_boundary' |  C_outside   ]
        
        the covaraince matrix used on the entire domain.
        
        This routine returns  C_inside * f.
        '''
        g = self.pad( f )
        
        cov_g = lap.fourier_multiplier( g, self.cov_eigs )        
    
        return self.project( cov_g )
        
    def apply_inv_schur_comp( self, f ):
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
        g = self.pad( f )
    
        cov_g = lap.fourier_multiplier( g, self.prec_eigs )
      
        return self.project( cov_g )
  
    def apply_precision( self, f, M=None, callback=None ):
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
        
        x_0 = np.zeros( self.small_domain )
        f = f.ravel()
        A = la.LinearOperator( (len(f),len(f)), matvec=self.apply_covariance )
        
        return la.cg( A,
                      f, 
                      x0=x_0,
                      tol=self.eps,
                      maxiter=None,
                      xtype=None, 
                      M=M,
                      callback=callback)[0]
    

if __name__ == "__main__":
    
    def callback(x):
        global count
        count = count + 1
    ran = range(4,32)  

    print( "Inverse Schur preconditioner" )
    for n in ran:
        
        count = 0
        par = Parameters(size = 2**n)
        f = np.sin( 12 * par.inside ) + np.cos( 17 * par.inside ) 
        cov_f = par.apply_covariance( f )
        shap = ( len(par.inside), len(par.inside) )
        M = Linear(shap, par.apply_inv_schur_comp )
        par.apply_precision( cov_f,
                             M=M,
                             callback=callback )
        print( str(2**n) + " " + str(count) )
        


    print( "No preconditioner" )
    for n in ran:
        
        count = 0
        par = Parameters(size = 2**n)
          
        f = np.sin( 12 * par.inside ) + np.cos( 17 * par.inside ) 
        cov_f = par.apply_covariance( f )
        par.apply_precision( cov_f,
                             callback=callback )
        print( str(2**n) + " " + str(count) )
    

    # Description of covariance operator
    # cov_str = "$(  %.2f \Delta +  %.2fI)^{ %.2f}$" % (
    #     par.alpha-1, par.alpha, par.power)
    # reconstruct_str = "$(  %.2f \Delta +  %.2fI)^{ %.2f} \circ $ " % (
    #     par.alpha-1, par.alpha, -par.power)
    # reconstruct_str = reconstruct_str + cov_str
  
    # # Plot eigenvalues
    # lin_op = la.LinearOperator(
    #     ( small_domain,small_domain ) , lambda v: apply_covariance( v, params ) )
    # eigs = la.eigs( lin_op, k = 500, return_eigenvectors = False ) 
    # eigs = np.sort( eigs )
    # eigs = eigs[::-1]
    # k = np.arange( 1, len( eigs ) + 1) 
    # powa = 1.6
    # k = k**powa
    # eigs = eigs * k

    # plt.plot( eigs, 'ro' )    
    # plt.xlabel('$k$')
    # title = 'Eigenvalues of covariance after cancelling their decay'
    # plt.ylabel('$\lambda_k \cdot k^{' + str(powa) + '}$')
    # plt.title(title)

    # # Tweak spacing to prevent clipping of ylabel
    # plt.subplots_adjust(left=0.15)
    # plt.savefig( "eigenvalues.png" )
    # plt.close()

    # Sample from the gaussian ##############
    # bound = 0
    # for i in range( 20 ):
    #     smp = sample( params )
    #     bound = max( bound, np.max( np.abs( smp ) ) ) 
        
    #     # plot - boring...
    #     plt.plot( inside, project(smp,params) )
    # axes = plt.gca()
    # axes.set_ylim( [-1.2 * bound, 1.2 * bound ] )
    # axes.set_xlim( [ 0.25,0.75 ] )
    # plt.title( "Samples from Gaussian with zero mean and covaraince " + cov_str +
    #            ".\nBig domain is $[0,1]$ interval. Subdomain is $[0.25,0.75]$." )
    # plt.savefig( "Samples.png" )
    # plt.close()
    
                                    
         
    # Plot shit 
    # plt.plot( par.inside, reconstruct_f, color = "b" , label = "reconstructed u" )
    # plt.plot( par.inside, cov_f        , color = "r" , label = cov_str + "u"     )
    # plt.plot( par.inside, f            , color = "g" , label = "u"               )
    # axes = plt.gca()
    # axes.set_xlim( [ 0.25,0.75 ] )
    # plt.title( "Apply " + cov_str + "\nand its inverse to reconstruct to a function." )
    # plt.legend( loc=2, prop={'size':6} )
    # plt.savefig( "Apply Covariance and precision.png" )
    # plt.close()

    # Plot empirical covaraince matrix #############
    # num_samples = 1000
    # smp = par.sample( num_samples )
    
    # big_cov = np.dot( smp, smp.T ) / num_samples
    
    # plt.imshow( big_cov )
    # plt.colorbar()
    # plt.title( "Covariance " + cov_str +"\non $[0,1]$ using "
    #            +str(num_samples) + " samples." )
    # plt.savefig( "Big Covariance.png" )
    # plt.close()

    # smp = par.project( smp ) 
    # rest_cov = np.dot( smp, smp.T ) / num_samples
    
    # plt.imshow( rest_cov )
    # plt.colorbar()
    # plt.title( "Covariance " + cov_str +"\non $[0.25,0.75]$ using "
    #            +str(num_samples) + " samples." )
    # plt.savefig( "Restricted Covariance.png" )
    # plt.close()
    
