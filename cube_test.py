#!/usr/bin/python
import scipy as sp
import numpy as np

from dolfin import *

import container
import betas
import helper
from helper import dic as dic

def enum( x0,x1,x2, kappa, n, reg ):
    ra = np.sqrt( x0*x0 + x1*x1 + x2*x2 ) + reg
    kappara = kappa * ra
    tmp = ( 2.0+np.power(kappara,-1) ) * np.exp( -kappara ) * np.power( kappara, -1.5 ) * sp.special.kv( 0.5, kappara ) * x0
    return kappa**2 * np.sum(tmp) / n / n / n

def denom( x0,x1,x2, kappa, n, reg ):
    ra = np.sqrt( x0*x0 + x1*x1 + x2*x2) + reg
    kappara = kappa * ra 
    tmp =   np.power( kappara, -0.5 ) * sp.special.kv( 0.5, kappara ) * np.exp( -kappara )
    return 2.0 * np.sum(tmp) / n / n / n

reg = 1e-12
n =  17 * 19
x0 = np.linspace(   0, 1.0, n, endpoint = False )   
x1 = np.linspace( -.5,  .5, n, endpoint = False )
x2 = np.linspace( -.5,  .5, n, endpoint = False )

X0, X1, X2 = np.meshgrid( x0, x1, x2 )
alpha = 1.

container = container.Container( "cube",
                                 dic["cube"](),
                                 alpha )
kappa = container.kappa
                                 

fe_beta  = betas.BetaCubeAdaptive( container )
fe_beta  = fe_beta(0.0,0.5,0.5)

nx_enum  = enum (X0, X1, X2, kappa, n, reg )
nx_denom = denom(X0, X1, X2, kappa, n, reg )

print "Cube fenics  beta  = " + str(-fe_beta[0] )
print "Cube numerix beta  = " + str( nx_enum / nx_denom )

