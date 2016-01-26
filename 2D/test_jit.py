#!/usr/bin/python
from dolfin import *
from scipy import special as sp
import numpy as np
import parameters

x  = np.array( [-1.2 , -1.0 ] )
y  = np.array( [0.34 ,  1.1 ] )

mesh_obj = UnitSquareMesh(5,5)
kappa = 1.27
dim = 2
nu = 1
container = parameters.Container( mesh_obj, kappa, dim, nu )

beta = parameters.Robin( container, param = "hom_beta" )
beta.update( x )

jit = np.array( [
    [ beta.mat11(y) , beta.mat12(y)  ],
    [ beta.mat12(y) , beta.mat22(y)  ]
] )
pyt = container.mat(x,y)
assert np.all( abs( jit - pyt ) < 1E-10 )



g = parameters.Robin( container, param = "g" )
g.update( x )

jit = np.array( [
    [ g.rhs11(y) , g.rhs12(y) ],
    [ g.rhs21(y) , g.rhs22(y) ]
] )
pyt = container.rhs(x,y)
assert np.all( abs( jit - pyt ) < 1E-10 )
