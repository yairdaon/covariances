#!/usr/bin/python
import scipy as sp
import numpy as np

from dolfin import *

import container
import betas2D
import helper

def imp_enum( x0,x1, kappa, n ):
    ra = np.sqrt( x0*x0 + x1*x1 ) + 1e-13
    kappara = kappa * ra 
    tmp = np.power( kappara, -1 ) * sp.special.kv( 0.0, kappara ) * sp.special.kv( 1.0, kappara ) * x0 
    return  kappa**2 * np.sum(tmp) / n**2

def imp_denom( x0,x1, kappa, n ):
    ra = np.sqrt( x0*x0 + x1*x1 ) + 1e-13
    kappara = kappa * ra 
    tmp = np.power( sp.special.kv( 0.0, kappara ), 2 )
    return np.sum(tmp) / n**2

def mix_enum( x0,x1, kappa, n ):
    ra = np.sqrt( x0*x0 + x1*x1 ) + 1e-13
    kappara = kappa * ra
    k0 = sp.special.kv( 0.0, kappara )
    k1 = sp.special.kv( 1.0, kappara )
    tmp = (   np.power(k0,2)  +  np.power(k1,2)   ) * x0 
    return kappa**2 * np.sum(tmp) / n**2

def mix_denom( x0,x1, kappa, n ):
    ra = np.sqrt( x0*x0 + x1*x1 ) + 1e-13
    kappara = kappa * ra 
    tmp =   kappara * sp.special.kv( 0.0, kappara ) * sp.special.kv( 1.0, kappara )
    return 2.0 * np.sum(tmp) / n**2  

n = 13 * 17 * 17
x0 = np.linspace(   0, 1.0, n, endpoint = False )   
x1 = np.linspace( -.5,  .5, n, endpoint = False )
X0, X1 = np.meshgrid( x0, x1 )

kappa = 5.0
                  
mesh_obj = UnitSquareMesh( n, n )

container = container.Container( "square",
                                 mesh_obj,
                                 kappa ) # == kappa == Killing 

fe_imp_denom = betas2D.IntegratedExpression( container, "imp_denom" )(0.0,0.5)
fe_imp_enum0 =-betas2D.IntegratedExpression( container, "imp_enum0" )(0.0,0.5)
fe_imp_enum1 = betas2D.IntegratedExpression( container, "imp_enum1" )(0.0,0.5)

fe_mix_denom = betas2D.IntegratedExpression( container, "mix_denom" )(0.0,0.5)
fe_mix_enum0 =-betas2D.IntegratedExpression( container, "mix_enum0" )(0.0,0.5)
fe_mix_enum1 = betas2D.IntegratedExpression( container, "mix_enum1" )(0.0,0.5)

fe_imp_beta  = betas2D.Beta( container, "imp" )(0.0,0.5)
fe_mix_beta  = betas2D.Beta( container, "mix" )(0.0,0.5)

nx_imp_enum  = imp_enum (X0,X1,kappa,n)
nx_imp_denom = imp_denom(X0,X1,kappa,n)
nx_mix_enum  = mix_enum (X0,X1,kappa,n)
nx_mix_denom = mix_denom(X0,X1,kappa,n)

file_name = "../PriorCov/data/square/pointwise.txt"
helper.empty_file( file_name )
open( file_name, "a" ).write( "Kappa                  = " + str( kappa         ) + "\n" )

open( file_name, "a" ).write(  "Improper fenics  enum0 = " + str( fe_imp_enum0 ) + "\n" )     
open( file_name, "a" ).write(  "Improper numerix enum0 = " + str( nx_imp_enum  ) + "\n" )
open( file_name, "a" ).write(  "Improper fenics  denom = " + str( fe_imp_denom ) + "\n" )     
open( file_name, "a" ).write(  "Improper numerix denom = " + str( nx_imp_denom ) + "\n" )

open( file_name, "a" ).write(  "Mixed    fenics  enum0 = " + str( fe_mix_enum0 ) + "\n" )
open( file_name, "a" ).write(  "Mixed    numerix enum0 = " + str( nx_mix_enum  ) + "\n" )
open( file_name, "a" ).write(  "Mixed    fenics  denom = " + str( fe_mix_denom ) + "\n" )
open( file_name, "a" ).write(  "Mixed    numerix denom = " + str( nx_mix_denom ) + "\n" )

open( file_name, "a" ).write(  "Improper fenics  beta  = " + str(-fe_imp_beta[0]             ) + "\n" )
open( file_name, "a" ).write(  "Improper numerix beta  = " + str( nx_imp_enum / nx_imp_denom ) + "\n" )
open( file_name, "a" ).write(  "Mixed    fenics  beta  = " + str(-fe_mix_beta[0]             ) + "\n" )
open( file_name, "a" ).write(  "Mixed    numerix beta  = " + str( nx_mix_enum / nx_mix_denom ) + "\n" )

