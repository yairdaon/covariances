import numpy as np
from scipy import special as sp

# from dolfin import *
# import instant 

import container
import betas
import helper
from helper import dic as dic

print "until now you were just waiting for imports!!!"


############################################
# Test the cubature file and compilation ###
############################################
if False:
    
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
if False:
    def enum2D( x0,x1, kappa, n ):
        ra = np.sqrt( x0*x0 + x1*x1 ) + 1e-13
        kappara = kappa * ra
        k0 = sp.kv( 0.0, kappara )
        k1 = sp.kv( 1.0, kappara )
        tmp  = np.power(k0,2)  +  np.power(k1,2)
        return ( kappa * np.sum(tmp*x0) / n**2 , kappa * np.sum(tmp*x1) / n**2 )

    def denom2D( x0,x1, kappa, n ):
        ra = np.sqrt( x0*x0 + x1*x1 ) + 1e-13
        kappara = kappa * ra 
        tmp = ra * sp.kv( 0.0, kappara ) * sp.kv( 1.0, kappara )
        return 2.0 * np.sum(tmp) / n**2  


    n = 17 * 13 * 17                
    dic["square"].x = dic["square"].y = 5
    
    x0 = np.linspace(   0, 1.0, n, endpoint = False )   
    x1 = np.linspace( -.5,  .5, n, endpoint = False )
    X0, X1 = np.meshgrid( x0, x1 )

    mesh_obj = helper.dic["square"]()
    alpha = 25.
    cot = container.Container( "square",
                               mesh_obj,
                               alpha )

    cb_beta  = betas.Beta2DAdaptive( cot )(0.0,0.5)
    
    denom = denom2D( X0, X1, cot.kappa, n )
    enum  = np.array( enum2D( X0, X1, cot.kappa, n ) )  
    nx_beta = enum / denom
    
    fe_beta  = betas.Beta2D( cot )(0.0,0.5)

    print "Cubature 2D: " + str( -cb_beta )
    print "Numerix 2D:  " + str(  nx_beta )  
    print "FEniCS 2D:   " + str( -fe_beta ) 


############################################
# Now test in 3D ###########################
############################################
print "3D tests!"
n = 367
x0 = np.linspace(   0, 1.0, n, endpoint = False )   
x1 = np.linspace( -.5,  .5, n, endpoint = False ) 
x2 = np.linspace( -.5,  .5, n, endpoint = False )

X0, X1, X2 = np.meshgrid( x0, x1, x2 )
print "Grid ready!"

def enum3D( x0,x1,x2, kappa, n ):
    ra = np.sqrt( x0*x0 + x1*x1 + x2*x2 ) + 1e-9
    kappara = kappa * ra
    
    Khalf = sp.kv( 0.5, kappara )
    expon = np.exp( -kappara )

    tot = kappa * Khalf * expon * (2+1/kappara) * np.power( ra, -1.5 )
    
    tmp0 = tot * x0 
    tmp1 = tot * x1
    tmp2 = tot * x2
    
    return ( np.sum(tmp0)/n**3, np.sum(tmp1)/n**3, np.sum(tmp2)/n**3 )
    
def denom3D( x0,x1,x2, kappa, n ):
    ra = np.sqrt( x0*x0 + x1*x1 + x2*x2) + 1e-9
    kappara = kappa * ra 
   
    Khalf = sp.kv( 0.5, kappara )
    expon = np.exp( -kappara )

    tmp = Khalf * expon / np.sqrt( ra )
    return 2.0 * np.sum(tmp) / n**3  


print "make the mesh and container..."
alpha = 25.0
mesh_obj = helper.get_refined_mesh( "cube", 
                                    89,
                                    nor = 1, 
                                    tol = 0.3,
                                    factor = 0.5,
                                    greens = True )

cot = container.Container( "cube",
                           mesh_obj,
                           alpha ) 
print "Mesh and container ready!!"
kappa = cot.kappa

cb_beta_3d  = betas.BetaCubeAdaptive( cot )(0.0,0.502,0.502)

print "Naive integration..."
nx_denom_3d = denom3D( X0, X1, X2, kappa, n )
nx_enum_3d  = enum3D ( X0, X1, X2, kappa, n )
nx_beta_3d  = np.array( nx_enum_3d ) / nx_denom_3d
print "Done!!!"

print "FEniCS integration..."
fe_beta_3d = betas.Beta3D( cot )(0.0, 0.502, 0.502)
#fe_denom   = betas.IntegratedExpression( cot, "denom_3d" )(0.0, 0.502, 0.502)
print "Done!!!"

print "Cubature 3D: " + str( -cb_beta_3d )
print "Numerix  3D: " + str(  nx_beta_3d )  
print "FEniCS   3D: " + str( -fe_beta_3d )  


















# file_name = "../PriorCov/data/square/pointwise.txt"
# helper.empty_file( file_name )
# open( file_name, "a" ).write(  "Kappa                  = " + str( kappa        ) + "\n" )

# open( file_name, "a" ).write(  "Mixed    fenics  enum0 = " + str( fe_enum0 ) + "\n" )
# open( file_name, "a" ).write(  "Mixed    numerix enum0 = " + str( nx_enum  ) + "\n" )
# open( file_name, "a" ).write(  "Mixed    fenics  denom = " + str( fe_denom ) + "\n" )
# open( file_name, "a" ).write(  "Mixed    numerix denom = " + str( nx_denom ) + "\n" )

# open( file_name, "a" ).write(  "Mixed    fenics  beta  = " + str(-fe_beta[0]             ) + "\n" )
# open( file_name, "a" ).write(  "Mixed    numerix beta  = " + str( nx_enum / nx_denom ) + "\n" )

# print open( file_name, "r").read()

