import numpy as np
from scipy import special as sp
import time
import math

import container
import betas
import helper
from helper import dic as dic

############################################
# Test the cubature file and compilation ###
############################################
if True:
    
    def calculate( name, y, kappa, arr, tol): 
        
        if "simple" in name or "singular" in name:
            result = np.empty( 1 )
        else:
            result = np.empty( 3 )
    
        xpr = betas.generateInstant( name )
        xpr.integrateVector( y, kappa, arr, tol, result )  
    
        return result
    
    arr = np.array( [
        -3, 0, # a
        6, 0, # b
        0, 3  # c
    ] , dtype = np.float64 )
    tol = 1e-5
    val = calculate( "simple2D", np.zeros(2),  1.23, arr, tol)
    print "\n\nExample from some exam\n"
    print "Computed integral = "  + str (val[0]        )  
    print "Analytic solution = "  + str (13.5          )  
    print "Difference        = "  + str (val[0] - 13.5 )  
    print "Tolerance         = "  + str ( tol          )
    assert abs( val - 13.5 ) / 13.5 < tol

    arr = np.array( [
        0, 0, # a
        1, 0, # b
        1, 1 # c
        ], dtype = np.float64 )

    val = calculate( "singular2D",   np.zeros(2),  1.23, arr, tol)    
    print "\n\nExample I made\n"
    print "Computed integral = "  + str ( val[0]                   )  
    print "Analytic solution = "  + str ( .184014120655140         )  
    print "Difference        = "  + str (val[0] - .184014120655140 )  
    print "Tolerance         = "  + str ( tol                      )  
    assert abs( val -.184014120655140 ) /.184014120655140 < tol


############################################
# Test the beta modules ####################
############################################
if True:
    print "2D tests!!"

    def enum2D( x0,x1, kappa, n, factor = 1 ):
        ra = np.sqrt( x0*x0 + x1*x1 ) + 1e-13
        kappara = kappa * ra
        k0      = sp.kn( 0, kappara )
        k1      = sp.kn( 1, kappara )
        tmp     = -kappa * factor * kappara * ( k0*k0 + k1*k1 ) / ra
    
        return (  np.sum(tmp*x0) / n**2 , np.sum(tmp*x1) / n**2 )

    def denom2D( x0,x1, kappa, n, factor  = 1 ):
        ra = np.sqrt( x0*x0 + x1*x1 ) + 1e-13
        kappara = kappa * ra 
        tmp = factor * kappara * sp.kv( 1.0, kappara ) * sp.kn( 0, kappara )
        return 2.0 * np.sum(tmp) / n**2  


    n = 17 * 13 * 17         
    x0 = np.linspace(   0, 1.0, n, endpoint = False )   
    x1 = np.linspace( -.5,  .5, n, endpoint = False )
    X0, X1 = np.meshgrid( x0, x1 )

    mesh_obj = helper.get_mesh( "square" , 50 ) 
    alpha = 15.
    cot = container.Container( "square",
                               mesh_obj,
                               alpha )
    
    cb_beta  = betas.Beta2DAdaptive( cot )(0.0,0.5)
    
    factor = cot.factor / 2. / math.pi 
    denom = denom2D( X0, X1, cot.kappa, n, factor = factor )
    enum  = np.array( enum2D( X0, X1, cot.kappa, n, factor = factor ) )  
    nx_beta = enum / denom
    
    fe_beta  = betas.Beta2D( cot )(0.0,0.5)

    rd_beta  = betas.Beta2DRadial( cot )(0.0, 0.5)

    print "Cubtur 2D: " + str( -cb_beta )
    print "Numrix 2D: " + str( -nx_beta )  
    print "FEniCS 2D: " + str( -fe_beta ) 
    print "Radial 2D: " + str( -rd_beta )
      
############################################
# Now test in 3D ###########################
############################################
if True:
    print "3D tests!"

    def enum3D( x0,x1,x2, kappa, n ):
        ra = np.sqrt( x0*x0 + x1*x1 + x2*x2 ) + 1e-9
        kappara = kappa * ra
        
        Khalf = sp.kv( 0.5, kappara )
        expon = np.exp( -kappara )
        
        tot = -kappa * Khalf * expon * (2+1/kappara) * np.power( ra, -1.5 )
    
        tmp0 = tot * x0 
        tmp1 = tot * x1
        tmp2 = tot * x2
    
        return ( np.sum(tmp0), np.sum(tmp1), np.sum(tmp2) )
    
    def denom3D( x0,x1,x2, kappa, n ):
        ra = np.sqrt( x0*x0 + x1*x1 + x2*x2) + 1e-9
        kappara = kappa * ra 
   
        Khalf = sp.kv( 0.5, kappara )
        expon = np.exp( -kappara )

        tmp = Khalf * expon / np.sqrt( ra )
        return 2.0 * np.sum(tmp)  

    
    print "make the mesh and container..."
    alpha = 25.0
    mesh_obj = helper.get_refined_mesh( "cube", 
                                        77,
                                        nor = 0, 
                                        tol = 0.5,
                                        factor = 0.5,
                                        greens = True )

    cot = container.Container( "cube",
                               mesh_obj,
                               alpha ) 
    print "Mesh and container ready!!"

    n = 555
    x0 = np.linspace(   0, 1.0, n, endpoint = False )   
    x1 = np.linspace( -.5,  .5, n, endpoint = False ) 
    x2 = np.linspace( -.5,  .5, n, endpoint = False )

    X0, X1, X2 = np.meshgrid( x0, x1, x2 )
    print "Grid ready!"


    kappa = cot.kappa
        
    print "Cubature..."
    b = betas.BetaCubeAdaptive( cot )
    start_time = time.time()
    cb_beta_3d = b(0.0,0.5,0.5)
    print "Run time: " + str( time.time() - start_time )

    print "Naive integration..."
    start_time = time.time()
    nx_denom_3d = denom3D( X0, X1, X2, kappa, n )
    nx_enum_3d  = enum3D ( X0, X1, X2, kappa, n )
    nx_beta_3d  = np.array( nx_enum_3d ) / nx_denom_3d
    print "Run time: " + str( time.time() - start_time )
    
   
    print "FEniCS integration..." 
    b = betas.Beta3D( cot )
    start_time = time.time()
    fe_beta_3d = b(0.0, 0.5, 0.5 )
    run = time.time() - start_time
    print "Run time: " + str( run )

    print "Radial integration..." 
    b = betas.Beta3DRadial( cot )
    start_time = time.time()
    rd_beta_3d = b(0.0, 0.5, 0.5 )
    run = time.time() - start_time
    print "Run time: " + str( run )

    print "Cubtur 3D: " + str( -cb_beta_3d ) 
    print "Numrix 3D: " + str( -nx_beta_3d )   
    print "FEniCS 3D: " + str( -fe_beta_3d ) 
    #print "FEniCS Err : " + str( np.abs( (fe_beta_3d-cb_beta_3d)/cb_beta_3d ) )
    print "Radial   3D: " + str( -rd_beta_3d )
    #print "Radial Err : " + str( np.abs( (rd_beta_3d-cb_beta_3d)/cb_beta_3d ) )

