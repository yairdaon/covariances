import numpy as np
import scipy as sp

from dolfin import *
import instant 

import container
import betas
import helper
from helper import dic as dic

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


# arr = np.array( [
#         0, 0, # a
#         1, 0, # b
#         1, 1  # c
#         ], dtype = np.float64 )
# tol  =1e-4
# val = calculate( "mix2D",  np.array( [0.2,0], dtype = np.float64 ),  1.23, arr, tol)    
# print "\n\nDenominator\n"
# print "Computed integral = "  + str ( val                   )  
# print "Analytic solution = "  + str ( .184014120655140        )  
# print "Difference        = "  + str (val - .184014120655140 )  
# print "Tolerance         = "  + str ( tol                     )  

# -----------------------------------------------------------------------------------


# def imp_enum( x0,x1, kappa, n ):
#     ra = np.sqrt( x0*x0 + x1*x1 ) + 1e-13
#     kappara = kappa * ra 
#     tmp = np.power( kappara, -1 ) * sp.special.kv( 0.0, kappara ) * sp.special.kv( 1.0, kappara ) * x0 
#     return  kappa**2 * np.sum(tmp) / n**2

# def imp_denom( x0,x1, kappa, n ):
#     ra = np.sqrt( x0*x0 + x1*x1 ) + 1e-13
#     kappara = kappa * ra 
#     tmp = np.power( sp.special.kv( 0.0, kappara ), 2 )
#     return np.sum(tmp) / n**2

def enum2D( x0,x1, kappa, n ):
    ra = np.sqrt( x0*x0 + x1*x1 ) + 1e-13
    kappara = kappa * ra
    k0 = sp.special.kv( 0.0, kappara )
    k1 = sp.special.kv( 1.0, kappara )
    tmp0 = (   np.power(k0,2)  +  np.power(k1,2)   ) * x0 
    tmp1 = (   np.power(k0,2)  +  np.power(k1,2)   ) * x1
    return ( kappa * np.sum(tmp0) / n**2 , kappa * np.sum(tmp1) / n**2 )

def denom2D( x0,x1, kappa, n ):
    ra = np.sqrt( x0*x0 + x1*x1 ) + 1e-13
    kappara = kappa * ra 
    tmp = ra * sp.special.kv( 0.0, kappara ) * sp.special.kv( 1.0, kappara )
    return 2.0 * np.sum(tmp) / n**2  


n = 17 * 13 * 17
dic["square"].x = dic["square"].y = 5

x0 = np.linspace(   0, 1.0, n, endpoint = False )   
x1 = np.linspace( -.5,  .5, n, endpoint = False )
X0, X1 = np.meshgrid( x0, x1 )

mesh_obj = helper.get_refined_mesh( "square",
                                    nor = 0,
                                    tol = 0.4,
                                    factor = 0.9,
                                    greens = True )
alpha = 25.
cot = container.Container( "square",
                           mesh_obj,
                           alpha ) # == kappa == Killing 

# fe_denom = betas.IntegratedExpression( cot, "denom" )(0.0,0.5)
# fe_enum0 =-betas.IntegratedExpression( cot, "enum0" )(0.0,0.5)
# #fe_enum1 = betas.IntegratedExpression( cot, "enum1" )(0.0,0.5)

cb_beta  = betas.Beta2DAdaptive( cot )(0.0,0.5)

denom = denom2D( X0, X1, cot.kappa, n )
enum  = np.array( enum2D( X0, X1, cot.kappa, n ) )  
nx_beta = enum / denom

# fe_beta  = betas.DeprecatedBeta( cot, "mix" )(0.0,0.5)

print "Cubature: " + str( -cb_beta )
print "Numerix:  " + str(  nx_beta )  
# print "FEniCS:   " + str( -fe_beta ) 

# nx_imp_enum  = imp_enum (X0,X1,kappa,n)
# nx_imp_denom = imp_denom(X0,X1,kappa,n)
# nx_enum  = enum (X0,X1,kappa,n)
# nx_denom = denom(X0,X1,kappa,n)

# file_name = "../PriorCov/data/square/pointwise.txt"
# helper.empty_file( file_name )
# open( file_name, "a" ).write(  "Kappa                  = " + str( kappa        ) + "\n" )

# open( file_name, "a" ).write(  "Improper fenics  enum0 = " + str( fe_imp_enum0 ) + "\n" )     
# open( file_name, "a" ).write(  "Improper numerix enum0 = " + str( nx_imp_enum  ) + "\n" )
# open( file_name, "a" ).write(  "Improper fenics  denom = " + str( fe_imp_denom ) + "\n" )     
# open( file_name, "a" ).write(  "Improper numerix denom = " + str( nx_imp_denom ) + "\n" )

# open( file_name, "a" ).write(  "Mixed    fenics  enum0 = " + str( fe_enum0 ) + "\n" )
# open( file_name, "a" ).write(  "Mixed    numerix enum0 = " + str( nx_enum  ) + "\n" )
# open( file_name, "a" ).write(  "Mixed    fenics  denom = " + str( fe_denom ) + "\n" )
# open( file_name, "a" ).write(  "Mixed    numerix denom = " + str( nx_denom ) + "\n" )

# open( file_name, "a" ).write(  "Improper fenics  beta  = " + str(-fe_imp_beta[0]             ) + "\n" )
# open( file_name, "a" ).write(  "Improper numerix beta  = " + str( nx_imp_enum / nx_imp_denom ) + "\n" )
# open( file_name, "a" ).write(  "Mixed    fenics  beta  = " + str(-fe_beta[0]             ) + "\n" )
# open( file_name, "a" ).write(  "Mixed    numerix beta  = " + str( nx_enum / nx_denom ) + "\n" )



# print open( file_name, "r").read()





# x0 = np.linspace(   0, 1.0, n, endpoint = False )   
# x1 = np.linspace( -.5,  .5, n, endpoint = False )
# x2 = np.linspace( -.5,  .5, n, endpoint = False )

# X0, X1, X2 = np.meshgrid( x0, x1, x2 )

# def enum3D( x0,x1,x2, kappa, n ):
#     ra = np.sqrt( x0*x0 + x1*x1 + x2*x2 ) + 1e-9
#     kappara = kappa * ra
    
#     Khalf = sp.special.kv( 0.5, kappara )
#     expon = np.exp( -kappara )

#     tot = kappa * Khalf * (2+1/kappara) * np.power( ra, -1.5 )
    
#     tmp0 = tot * x0 
#     tmp1 = tot * x1
#     tmp2 = tot * x2
    
#     return ( np.sum(tmp0) / n**2 , np.sum(tmp1) / n**2 , np.sum(tmp2) / n**2 )
    
# def denom3D( x0,x1,x2, kappa, n ):
#     ra = np.sqrt( x0*x0 + x1*x1 + x2*x2) + 1e-9
#     kappara = kappa * ra 
   
#     Khalf = sp.special.kv( 0.5, kappara )
#     expon = np.exp( -kappara )

#     tmp = Khalf * expon / np.sqrt( ra )
#     return 2.0 * np.sum(tmp) / n**2  



# alpha = 25.0
# mesh_obj = helper.get_refined_mesh( "cube",
#                                     nor = 0,
#                                     tol = 0.4,
#                                     factor = 0.9,
#                                     greens = False )

# cot = container.Container( "cube",
#                            mesh_obj,
#                            alpha ) 
# kappa = cot.kappa

# cb_beta  = betas.BetaCubeAdaptive( cot )(0.0,0.5,0.0)

# denom = denom3D( X0, X1, X2, kappa, n )
# nx_beta  = enum3D( X0, X1, X2, kappa, n )
# nx_beta = np.array( nx_beta) / denom

# # fe_beta  = betas.DeprecatedBeta( cot, "mix" )(0.0,0.5)

# print "Cubature: " + str( -cb_beta )
# print "Numerix:  " + str(  nx_beta )  
