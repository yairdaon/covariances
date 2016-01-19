#!/usr/bin/python
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import pdb

n =200
mesh_obj = UnitIntervalMesh( n )

V = FunctionSpace( mesh_obj, "CG", 1 )


# Choose kappa
kappa = Function( V ) 
kappa = kappa +1.45

# All the pbjects used in the variational formulation below
u = TrialFunction ( V )
v = TestFunction ( V )
f = Constant( 0.0 )
  
# Neumann Variational formulation
a = inner(grad(u), grad(v))*dx + kappa*u*v*dx
L = f*v*dx
A = assemble( a )
b = assemble( L )


delta = PointSource ( V, Point ( 0.5  ), 1. )
delta.apply( b )


# Solve the linear systems - Neumann.
sol = Function( V )
solve(A, sol.vector(), b, 'lu')
 
# Save to memory
file = File( "neumann.pvd")
file << sol

#Plot the solutions.
plot ( sol, interactive = True, title = "Neumann Solution" )
