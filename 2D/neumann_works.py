from dolfin import *
import numpy
import helper

kappa = 1.23
mesh_obj = Mesh( "dolfin_fine.xml" )
normal = FacetNormal( mesh_obj )

V = FunctionSpace ( mesh_obj, "CG", 1 )
u = TrialFunction ( V )
v = TestFunction ( V )
f = Constant( .0 )
  
beta = helper.Beta( kappa, mesh_obj, V )
      
# Sample variational forms
a = inner(grad(u), grad(v))*dx + kappa*kappa*u*v*dx
L = f*v*dx

A = assemble(a)
b = assemble(L)

# Modify the right hand side vector to account for point source(s)
delta = PointSource ( V, Point ( 0.6, 0.5 ), 10.0 )
delta.apply ( b )
gamma = PointSource ( V, Point( 0.45, 0.65 ), 10.0 )
gamma.apply( b )
eta = PointSource ( V, Point( 0.35, 0.35 ), 10.0 )
eta.apply( b )

#  Solve the linear system for u.
sol = Function( V )
solve(A, sol.vector(), b, 'lu')

    
# Plot the solution.
plot ( sol, interactive = True ) 
