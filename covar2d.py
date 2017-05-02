#!/usr/local/bin/python3
import numpy as np
from numpy.fft import rfft2 as rfft2
from numpy.fft import irfft2 as irfft2
from numpy.fft import rfftfreq as rfftfreq 
from numpy import dot as dot
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import scipy.sparse.linalg as la
import pdb
import math
import sys

def pcg(A, b, x, M, tol ):

    '''
    Jonathan Shewchuk's canned
    preconditioned CG from his famous 
    paper "Conjugate Gradient Method
    Without the Agonizing Pain".
    '''

    factor = int( math.sqrt(len(b)) ) + 1
    tol2 = tol*tol

    i = 0
    r = b - A(x)
    d = M(r)
    deltaNew = dot( r, d ) 
    delta0 = deltaNew
    while deltaNew > tol2 * delta0:
        q = A(d)
        alpha = deltaNew / dot( d, q )
        x = x + alpha * d

        if i % factor == 0:
            r = b - A(x)
        else:
            r = r - alpha * q

        s = M(r)

        deltaOld = deltaNew
        deltaNew = dot( r,s )
        beta = deltaNew / deltaOld
        d = s + beta * d
        i = i+1
    
    return x, i


class Parameters(object):
    
    def __init__( self, out, power, alpha=0.25, sigma=1 ):
       
        # Evenly numbered domains are your friend.
        assert( out % 4 == 0 )
       
        # Set the points
        self.out = out
        self.start =     out // 4
        self.stop  = 3 * out // 4 + 1
        self.ins   = self.stop - self.start

        grid = np.linspace( 0, 1, out, endpoint=False )
        self.outX, self.outY = np.meshgrid( grid, grid ) 
        
        self.insX = self.outX[ self.start:self.stop , self.start:self.stop ]
        self.insY = self.outY[ self.start:self.stop , self.start:self.stop ]

 
        # threshold for conjugate gradients
        self.eps = 1E-7
        
        # Alpha is the relative weight we give the 
        # identity operator. 
        assert( alpha > 0 and alpha < 1 )
        
        # What we get when twice differnetiating complex exponentials
        factor = 4 * np.pi * np.pi 
    
        # ????????????????????????????????
        eigs_x = np.arange( 0, out/2 + 1 ) 
        eigs_y = np.arange( 0, out ) 
        eigs_x, eigs_y = np.meshgrid( eigs_x, eigs_y )
        
        eigs = eigs_x * eigs_x + eigs_y * eigs_y
        eigs = alpha + eigs
        self.eigs      = sigma**2 * np.power( eigs, -power )


    def project( self, v ):
        '''
        Take a vector v of lenght N and return its projection
        to the first n coordinates.
        
        In Linear Algebraic terms, project can be thought
        of as a n x N matrix A s.t. A_{ij} = 1 iff i = j and 
        A_{ij} = 0 otherwise. It is the transpose of the 
        pad matrix.
        '''
        
        v = v.reshape( (self.out,self.out) )
        v = v[ self.start:self.stop, self.start:self.stop ]
        
        return v.ravel()

    def pad( self, v ):
        '''
        take a vector v of length n and pad it with zeros
        so that the resulting vector has length N.
        
        In Linear Algebraic terms, pad can be thought of as
        a N x n matrix A s.t. A_{ij} = 1 iff i = j and 
        A_{ij} = 0 otherwise. It is the transpose of the 
        projection matrix.
        '''    
        
        v = v.reshape( (self.ins,self.ins) )        
        u = np.zeros(  (self.out,self.out) )
        u[ self.start:self.stop, self.start:self.stop ] = v
        return u.ravel()
       
    def fourier_multiplier( self, f, powa ):
        '''
        use an operator that is a Fourier multiplier
        on either domain (dependeing on the shape of the input)
        
        powa is a function that takes an array of eigenvalues
        and raises it to the right power
        '''

        assert np.all( f.imag == 0 )
        
        #pdb.set_trace()
        if np.size(f) == self.ins**2:        
            dim = self.ins
            f = self.pad( f )
            
        elif np.size(f) == self.out**2:
            dim = self.out

        else:
            raise ValueError("Inconsistent dimensions for Fourier multiplier")
                       
        f = f.reshape( (self.out,self.out) )       
        f_hat = rfft2( f )
        f = irfft2( f_hat * powa(self.eigs) )
  
        assert np.linalg.norm( np.imag(f) ) < self.eps
        
        f = np.real( f ) 
        if dim == self.ins:
            return self.project( f.ravel() )
        else:
            return f.ravel()
        
    def sample( self ):
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
        Z = np.random.randn( self.out**2 )  
            
        # Apply covariance to power half. This samples
        # from our covaraince function.
        sample = self.fourier_multiplier( Z, np.sqrt )
        return self.project( sample )

    def apply_covariance( self, f ):
        ''' 
        Apply covariance in a subdomain.
        
        Define C_all =
        [ C_inside    |  C_boundary  ] 
        [ ------------|------------- ],
        [ C_boundary' |  C_outside   ]
        
        the covaraince matrix used on the entire domain.
        
        This routine returns  C_inside * f.
        '''
       
        cov = self.fourier_multiplier( f, lambda x: x )        
    
        return self.project( cov )
        
    def apply_inv_schur_comp( self, f ):
        ''' 
        Apply inverse of covariance's Schur complement.
        
        Define C_all =
        [ C_inside    |  C_boundary  ] 
        [ ------------|------------- ],
        [ C_boundary' |  C_outside   ]
        
        the covaraince matrix used on the entire domain.
        
        This routines returns  
        (C_in - C_bd * C_out^{-1} * C_bd' )^{-1} * f. 
        This is the inverse of the schur complement of
        C_in applied to f.
        '''
        #pdb.set_trace()
        
        # M stands for the preconditioner
        return  self.fourier_multiplier( f, lambda x: 1. / x )
        
    def invert_fourier_multiplier( self, f, powa, M=lambda x: x, ):
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
        
        x = np.zeros( self.ins**2 )
        f = f.ravel()
        A = lambda f: self.fourier_multiplier( f, powa ) 

        return pcg(A, # linear operator
                   f, # right hand side
                   x, # starting point
                   M, # preconditioner
                   self.eps # (realtive) tolerance
               )
        
    
def make_plots( n ):
    
    par1 = Parameters(2**n, 2)
    f1 = par1.sample()
    f1 = f1.reshape( (par1.ins, par1.ins) ) 
  
    par2 = Parameters(2**n, 1.8)
    f2 = par2.sample()
    f2 = f2.reshape( (par2.ins, par2.ins) ) 
  
    par3 = Parameters(2**n, 1.6)
    f3 = par3.sample()
    f3 = f3.reshape( (par3.ins, par3.ins) ) 
        
    fig = plt.figure(figsize=(12,4))
        
    ax1 = fig.add_subplot(131)
    ax1.pcolormesh( par1.insX, par1.insY, f1 )
    ax1.set_xlim( (0.25,0.75) )
    ax1.set_ylim( (0.25,0.75) )
    ax1.set_title( "$p=2$" )
        
    ax2 = fig.add_subplot(132)
    ax2.pcolormesh( par2.insX, par2.insY, f2 )
    ax2.set_xlim( (0.25,0.75) )
    ax2.set_ylim( (0.25,0.75) )
    ax2.set_title( "$p=1.8$" )
    
    ax3 = fig.add_subplot(133)
    ax3.pcolormesh( par3.insX, par3.insY, f3 )
    ax3.set_xlim( (0.25,0.75) )
    ax3.set_ylim( (0.25,0.75) )
    ax3.set_title( "$p=1.6$" )
    plt.savefig('../thesis/data/fft2d/samples.png', bbox_inches='tight')
    plt.savefig('../thesis/data/fft2d/samples.pdf', bbox_inches='tight')
    #plt.show()     

def run_one( n, power, prec, filename ):       

    par = Parameters(2**n, power = power)

    # Create the preconditioner
    if prec:
        M = par.apply_inv_schur_comp
    else:
        M = lambda x: x


    count = 0
    countSqr = 0
    tot = 0
    while True:            
            
        f = par.sample()
        rec_f, m = par.invert_fourier_multiplier( f,
                                                  np.sqrt,
                                                  M=M )
        count = count + m
        countSqr = countSqr + m*m
        
        tot = tot + 1
        
        avg = count / tot
        std = math.sqrt(countSqr/tot - (count/tot)**2)

        if tot % 10 == 0 and tot > 50 and std/avg < 0.1:
            break
    dat = str(4**n) + " " + str(avg) + "\n"       
    open( filename, "a").write(dat)
                
if __name__ == "__main__":

    make_plots( 12 )
    powers = [1.6, 1.8, 2.]
    ran = range(3,10)
    
  
    dest = "../thesis/data/fft2d/"
    fin = ".txt"
    for n in ran:
        
        print( "Create files... n = " + str(n) )
        
        if n == 3:
            for prec in ["NoPrec", "InvSchur"]:
                for power in powers:
                    filename = dest + prec + str(int(10*power)) + fin
                    open(filename, 'w+').close()

        print( "Made files!!! n = " + str(n ) )
        for power in powers:
            filename = dest +  "NoPrec"  + str(int(10*power)) + fin
            print( "Running p = " + str(power) + " with no preconditioner." )
            run_one( n, power, False, filename )
            
            filename = dest +  "InvSchur"  + str(int(10*power)) + fin
            print( "Running p = " + str(power) + " with preconditioner." )
            run_one( n, power, True, filename )  
            
        print( "Finished n = " + str(n) )
    

   
