#!/usr/bin/python
from dolfin import *
from scipy import special as sp
import numpy as np
import helper
import matern
x  = np.array( [1.2  , -1.0 ] )
y  = np.array( [0.34 , 1.1  ] )
r = np.linalg.norm( x-y )

mesh_obj = UnitSquareMesh(5,5)
kappa = 1.27
container = helper.Container( mesh_obj, kappa, 2,  deg=1 )

robin = helper.Robin( container )


robin.update_x( x )

jit = robin.mat( y )
pyt = container(x,y)
print jit
print pyt

assert abs( jit - pyt ) < 1E-10

######
jit = robin.gradG( y )
pyt = -kappa * sp.kn( 1 , kappa * r ) * (x-y) / r
print jit
print pyt

assert np.all( np.abs( jit - pyt ) < 1E-10 )
