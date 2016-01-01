#!/usr/bin/python
from dolfin import *
from scipy import special as sp
import numpy as np
import helper

x  = np.array( [1.2  , -1.0 ] )
y  = np.array( [0.34 , 1.1  ] )
r = np.linalg.norm( x-y )

mesh_obj = UnitSquareMesh(5,5)
kappa = 1.27
beta = helper.Beta( kappa, mesh_obj, 1)


beta.update_x( x )

jit = beta.G( y )[0]
pyt = sp.kn(0, kappa*r )
print jit
print pyt

assert abs( jit - pyt ) < 1E-10

######
jit = beta.gradG( y )
pyt = -kappa * sp.kn( 1 , kappa * r ) * (x-y) / r
print jit
print pyt

assert np.all( np.abs( jit - pyt ) < 1E-10 )
