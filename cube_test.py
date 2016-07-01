#!/usr/bin/python
import scipy as sp
import numpy as np

from dolfin import *

import container
import mixed3D

def mix_enum( x0,x1,x2, kappa, n, reg ):
    ra = np.sqrt( x0*x0 + x1*x1 + x2*x2 ) + reg
    kappara = kappa * ra
    tmp = ( 2.0+np.power(kappara,-1) ) * np.exp( -kappara ) * np.power( kappara, -1.5 ) * sp.special.kv( 0.5, kappara ) * x0
    return kappa**2 * np.sum(tmp) / n / n / n

def mix_denom( x0,x1,x2, kappa, n, reg ):
    ra = np.sqrt( x0*x0 + x1*x1 + x2*x2) + reg
    kappara = kappa * ra 
    tmp =   np.power( kappara, -0.5 ) * sp.special.kv( 0.5, kappara ) * np.exp( -kappara )
    return 2.0 * np.sum(tmp) / n / n / n

reg = 1e-13
n = 13 * 17 * 19
x0 = np.linspace(   0, 1.0, n, endpoint = False )   
x1 = np.linspace( -.5,  .5, n, endpoint = False )
x2 = np.linspace( -.5,  .5, n, endpoint = False )

X0, X1, X2 = np.meshgrid( x0, x1, x2 )
kappa = 1.
                  
mesh_obj = helper.get_mesh( )

container = container.Container( "cube",
                                 mesh_obj,
                                 kappa ) # == kappa == Killing rate

                                 

fe_mix_beta  = mixed3D.Mixed( container, reg = reg )
fe_mix_beta  = fe_mix_beta(0.0,0.5,0.5)

nx_mix_enum  = mix_enum (X0, X1, X2, kappa, n, reg )
nx_mix_denom = mix_denom(X0, X1, X2, kappa, n, reg )

print "Mixed Cube fenics  beta  = " + str(-fe_mix_beta[0] )
print "Mixed Cube numerix beta  = " + str( nx_mix_enum / nx_mix_denom )

