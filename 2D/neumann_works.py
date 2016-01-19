#!/usr/bin/python
from dolfin import *
import numpy
import helper

kappa = 2.03
mesh_obj = Mesh( "dolfin_coarse.xml" )
normal = FacetNormal( mesh_obj )

V = FunctionSpace ( mesh_obj, "CG", 1 )
u = TrialFunction ( V )
v = TestFunction ( V )
f = Constant( .0 )
      
# Sample variational forms
a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx
L = f*v*dx

A = assemble(a)
b = assemble(L)

# Modify the right hand side vector to account for point source(s)
delta = PointSource ( V, Point ( 0.45 , 0.65  ), 1. )
delta.apply ( b )

#  Solve the linear system for u.
sol1 = Function( V )
solve(A, sol1.vector(), b, 'lu')

L = sol1*v*dx
b = assemble(L)


    
# Plot the solution.
plot ( sol1, interactive = True ) 

# Solve again
sol2 = Function( V )
solve(A, sol2.vector(), b, 'lu')

    
# Plot the solution.
plot ( sol2, interactive = True ) 
