#!/usr/bin/python
from dolfin import *
from scipy import special as sp
import numpy as np
import parameters
import helper

x  = np.array( [-1.2 , -1.0 ] )
y  = np.array( [0.34 ,  1.1 ] )

mesh_obj = UnitSquareMesh(5,5)
kappa = 1.27
dim = 2.0
nu = 1.0

container = parameters.Container( "square",
                                  mesh_obj,
                                  kappa,
                                  dim,
                                  nu,
                                  1 )

mat11 = container.generate( "mat11" )
helper.update_x_xp( x, mat11 )
assert abs( mat11(y) - container.mat11( x,y ) ) < 1e-9

rhs11 = container.generate( "rhs11" )
helper.update_x_xp( x, rhs11 )
assert abs( rhs11(y) - container.rhs11( x,y ) ) < 1e-9


rhs12 = container.generate( "rhs12" )
helper.update_x_xp( x, rhs12 )
assert abs( rhs12(y) - container.rhs12( x,y ) ) < 1e-9
